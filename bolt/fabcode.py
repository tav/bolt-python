# Changes to this file by The Bolt Authors are in the Public Domain.
# See the Bolt UNLICENSE file for details.

# Copyright (c) 2009-2011, Christian Vest Hansen and Jeffrey E. Forcier
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Support code derived from the Fabric codebase."""

import sys

from contextlib import closing, contextmanager, nested
from fnmatch import filter as fnfilter
from functools import wraps
from getpass import getpass
from glob import glob
from hashlib import sha1
from os import devnull, fdopen, getcwd, getuid, makedirs, remove, stat, walk
from os.path import abspath, basename, dirname, exists, expanduser, isabs, isdir
from os.path import join, split
from re import compile as compile_regex, findall
from select import select
from socket import error as socketerror, gaierror, timeout
from stat import S_ISDIR, S_ISLNK
from subprocess import PIPE, Popen
from tempfile import mkstemp
from textwrap import dedent
from threading import Thread
from time import sleep
from traceback import format_exc

# ------------------------------------------------------------------------------
# Platform Specific Imports
# ------------------------------------------------------------------------------

win32 = (sys.platform == 'win32')

if win32:
    import msvcrt
else:
    import fcntl
    import struct
    import termios
    import tty

# ------------------------------------------------------------------------------
# Some Constants
# ------------------------------------------------------------------------------

IO_SLEEP = 0.01

# ------------------------------------------------------------------------------
# Container Classes
# ------------------------------------------------------------------------------

class AttrDict(dict):
    """Dict subclass that allows for attribute access of key/values."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

class AliasDict(AttrDict):
    """Subclass that allows for "aliasing" of keys to other keys."""

    def __init__(self, mapping=None, aliases={}):
        self.update(mapping, aliases)

    def __setitem__(self, key, value):
        if key in self.aliases:
            for aliased in self.aliases[key]:
                self[aliased] = value
        else:
            return dict.__setitem__(self, key, value)

    def expand_aliases(self, keys):
        ret = []
        for key in keys:
            if key in self.aliases:
                ret.extend(self.expand_aliases(self.aliases[key]))
            else:
                ret.append(key)
        return ret

    def update(self, mapping=None, aliases={}):
        if mapping is not None:
            for key in mapping:
                dict.__setitem__(self, key, mapping[key])
        dict.__setattr__(self, 'aliases', aliases)

class EnvDict(AttrDict):
    """Environment dictionary object."""

# ------------------------------------------------------------------------------
# Global Objects
# ------------------------------------------------------------------------------

env = EnvDict()
output = AliasDict()

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def stringify_env_var(var):
    """Format the complete environment $VARIABLE setting string."""

    key = result = '$%s' % var
    for value, behaviour, sep in env.get(key, []):
        if behaviour == 'append':
            result = result + sep + '"' + value + '"'
        elif behaviour == 'prepend':
            result = '"' + value + '"' + sep + result
        else:
            result = '"' + value + '"'
    return "%s=%s" % (var, result)

def _get_system_username():
    """Return the current system user."""

    if not win32:
        import pwd
        return pwd.getpwuid(getuid())[0]
    else:
        import win32api
        import win32security
        import win32profile
        return win32api.GetUserName()

def _rc_path(rc_file='.bolt.yaml'):
    """Return the platform-specific path for $HOME/.bolt.yaml."""

    if not win32:
        return expanduser("~/" + rc_file)
    else:
        from win32com.shell.shell import SHGetSpecialFolderPath
        from win32com.shell.shellcon import CSIDL_PROFILE
        return "%s/%s" % (SHGetSpecialFolderPath(0,CSIDL_PROFILE), rc_file)

def _escape_split(sep, argstr):
    """Split string, allowing for escaping of the separator."""

    escaped_sep = r'\%s' % sep
    if escaped_sep not in argstr:
        return argstr.split(sep)

    before, _, after = argstr.partition(escaped_sep)
    startlist = before.split(sep)
    unfinished = startlist[-1]
    startlist = startlist[:-1]
    endlist = _escape_split(sep, after)
    unfinished += sep + endlist[0]
    return startlist + [unfinished] + endlist[1:]

# ------------------------------------------------------------------------------
# Authentication Support
# ------------------------------------------------------------------------------

def get_password():
    return env.passwords.get(env.host_string, env.password)

def set_password(password):
    env.password = env.passwords[env.host_string] = password

# ------------------------------------------------------------------------------
# Colours Support
# ------------------------------------------------------------------------------

def _bold_wrap_with(code):
    def inner(text):
        return "\033[1;%sm%s\033[0m" % (code, text)
    return inner

def _wrap_with(code):
    def inner(text, bold=False):
        c = code
        if bold:
            c = "1;%s" % c
        return "\033[%sm%s\033[0m" % (c, text)
    return inner

bold_blue = _bold_wrap_with('34')
bold_cyan = _bold_wrap_with('36')
bold_green = _bold_wrap_with('32')
bold_magenta = _bold_wrap_with('35')
bold_red = _bold_wrap_with('31')
bold_white = _bold_wrap_with('37')
bold_yellow = _bold_wrap_with('33')

blue = _wrap_with('34')
cyan = _wrap_with('36')
green = _wrap_with('32')
magenta = _wrap_with('35')
red = _wrap_with('31')
white = _wrap_with('37')
yellow = _wrap_with('33')

# ------------------------------------------------------------------------------
# Thread Handler
# ------------------------------------------------------------------------------

class ThreadHandler(object):
    """Wrapper around worker threads."""

    def __init__(self, name, callable, *args, **kwargs):
        self.exception = None
        def wrapper(*args, **kwargs):
            try:
                callable(*args, **kwargs)
            except BaseException:
                self.exception = sys.exc_info()
        thread = Thread(None, wrapper, name, args, kwargs)
        thread.setDaemon(True)
        thread.start()
        self.thread = thread

# ------------------------------------------------------------------------------
# Context Managers
# ------------------------------------------------------------------------------

@contextmanager
def char_buffered(pipe):
    """Force the local terminal ``pipe`` to be character, not line, buffered."""

    if win32 or env.get('disable_char_buffering', 0) or not sys.stdin.isatty():
        yield
    else:
        old_settings = termios.tcgetattr(pipe)
        tty.setcbreak(pipe)
        try:
            yield
        finally:
            termios.tcsetattr(pipe, termios.TCSADRAIN, old_settings)

def _set_output(groups, which):
    previous = {}
    for group in output.expand_aliases(groups):
        previous[group] = output[group]
        output[group] = which
    yield
    output.update(previous)

@contextmanager
def hide(*groups):
    """Hide output from the given ``groups``."""

    return _set_output(groups, False)

@contextmanager
def show(*groups):
    """Show output from the given ``groups``."""

    return _set_output(groups, True)

@contextmanager
def _setenv(**kwargs):
    previous = {}
    for key, value in kwargs.iteritems():
        if key in env:
            previous[key] = env[key]
        env[key] = value
    try:
        yield
    finally:
        env.update(previous)

def prefix(command):
    """Prefix ``run``/``sudo`` calls with the given ``command`` plus ``&&``."""

    return _setenv(command_prefixes=env.command_prefixes + [command])

def settings(*ctxmanagers, **env_values):
    """Nest the ``ctxmanagers`` and temporarily override the ``env_values``."""

    managers = list(ctxmanagers)
    if env_values:
        managers.append(_setenv(**env_values))
    return nested(*managers)

def _change_cwd(which, path):
    path = path.replace(' ', '\ ')
    if env.get(which) and not path.startswith('/'):
        new_cwd = env.get(which) + '/' + path
    else:
        new_cwd = path
    return _setenv(**{which: new_cwd})

def cd(path):
    """Prefix run/sudo/get/put calls to run in the given remote ``path``."""

    return _change_cwd('cwd', path)

def lcd(path):
    """Prefix local/get/put calls to run in the given local ``path``."""

    return _change_cwd('lcwd', path)

# ------------------------------------------------------------------------------
# Formatters
# ------------------------------------------------------------------------------

def indent(text, spaces=4, strip=False):
    """Return the ``text`` indented by the given number of ``spaces``."""

    if not hasattr(text, 'splitlines'):
        text = '\n'.join(text)
    if strip:
        text = dedent(text)
    prefix = ' ' * spaces
    output = '\n'.join(prefix + line for line in text.splitlines())
    output = output.strip()
    output = prefix + output
    return output

# ------------------------------------------------------------------------------
# Modified Output Utilities
# ------------------------------------------------------------------------------

def abort(msg):
    """Print the given ``msg`` and exit with status code 1."""

    if output.aborts:
        if env.colors:
            abort_color = env.color_settings['abort']
            print >> sys.stderr, abort_color("\nFatal error: " + str(msg))
            print >> sys.stderr, abort_color("\nAborting.")
        else:
            print >> sys.stderr, "\nFatal error: " + str(msg)
            print >> sys.stderr, "\nAborting."

    sys.exit(1)

def warn(msg):
    """Print the given warning ``msg``."""

    if output.warnings:
        msg = "\nWarning: %s\n" % msg
        if env.colors:
            print >> sys.stderr, env.color_settings['warn'](msg)
        else:
            print >> sys.stderr, msg

def puts(
    text, prefix=None, end="\n", flush=False, show_host=True, format=True
    ):
    """Print the given ``text`` within the constraints of the output state."""

    if output.user:
        if prefix:
            prefix = '[%s] ' % prefix
        else:
            prefix = ''
        if show_host and env.host_string:
            host_prefix = "[%s] " % env.host_string
        else:
            host_prefix = ''
        if env.colors:
            if prefix:
                prefix = env.color_settings['prefix'](prefix)
            if host_prefix:
                host_prefix = env.color_settings['host_prefix'](host_prefix)
        text = host_prefix + prefix + str(text) + end
        if format:
            text = text.format(**env)
        sys.stdout.write(text)
        if flush:
            sys.stdout.flush()

def fastprint(
    text, prefix=False, end="", flush=True, show_host=False, format=True
    ):
    """Like ``puts``, but defaults to printing immediately."""

    return puts(text, prefix, end, flush, show_host, format)

# ------------------------------------------------------------------------------
# Networking Support
# ------------------------------------------------------------------------------

try:
    import warnings
    warnings.simplefilter('ignore', DeprecationWarning)
    import paramiko as ssh
except ImportError:
    abort(
        """paramiko is a required module. Please install it:
        $ sudo easy_install paramiko
        """)

HOST_PATTERN = r'((?P<user>.+)@)?(?P<host>[^:]+)(:(?P<port>\d+))?'
HOST_REGEX = compile_regex(HOST_PATTERN)

class HostConnectionCache(dict):
    """Dict subclass allowing for caching of host connections/clients."""

    def __getitem__(self, key):
        user, host, port = normalize(key)
        real_key = join_host_strings(user, host, port)
        if real_key not in self:
            self[real_key] = connect(user, host, port)
        return dict.__getitem__(self, real_key)

    def __delitem__(self, key):
        return dict.__delitem__(self, join_host_strings(*normalize(key)))

CONNECTIONS = HostConnectionCache()

def default_channel():
    """Return a channel object based on ``env.host_string``."""

    return CONNECTIONS[env.host_string].get_transport().open_session()

def normalize(host_string, omit_port=False):
    """Normalize a ``host_string``, returning the explicit host, user, port."""

    if not host_string:
        return ('', '') if omit_port else ('', '', '')

    r = HOST_REGEX.match(host_string).groupdict()
    user = r['user'] or env.get('user')
    host = r['host']
    port = r['port'] or '22'
    if omit_port:
        return user, host

    return user, host, port

def denormalize(host_string):
    """Strip default values for the given ``host_string``."""

    r = HOST_REGEX.match(host_string).groupdict()
    user = ''
    if r['user'] is not None and r['user'] != env.user:
        user = r['user'] + '@'
    port = ''
    if r['port'] is not None and r['port'] != '22':
        port = ':' + r['port']
    return user + r['host'] + port

def join_host_strings(user, host, port=None):
    """Turn user/host/port strings into ``user@host:port`` combined string."""

    port_string = ''
    if port:
        port_string = ":%s" % port
    return "%s@%s%s" % (user, host, port_string)

def connect(user, host, port):
    """Create and return a new SSHClient connected to the given host."""

    client = ssh.SSHClient()

    if not env.disable_known_hosts:
        client.load_system_host_keys()

    if not env.reject_unknown_hosts:
        client.set_missing_host_key_policy(ssh.AutoAddPolicy())

    connected = False
    password = get_password()

    while not connected:

        try:
            client.connect(
                hostname=host,
                port=int(port),
                username=user,
                password=password,
                key_filename=env.key_filename,
                timeout=10,
                allow_agent=not env.no_agent,
                look_for_keys=not env.no_keys
            )
            connected = True
            return client
        # BadHostKeyException corresponds to key mismatch, i.e. what on the
        # command line results in the big banner error about man-in-the-middle
        # attacks.
        except ssh.BadHostKeyException:
            abort("Host key for %s did not match pre-existing key! Server's key was changed recently, or possible man-in-the-middle attack." % env.host)
        # Prompt for new password to try on auth failure
        except (
            ssh.AuthenticationException,
            ssh.PasswordRequiredException,
            ssh.SSHException
        ), e:
            # For whatever reason, empty password + no ssh key or agent results
            # in an SSHException instead of an AuthenticationException. Since
            # it's difficult to do otherwise, we must assume empty password +
            # SSHException == auth exception. Conversely: if we get
            # SSHException and there *was* a password -- it is probably
            # something non auth related, and should be sent upwards.
            if e.__class__ is ssh.SSHException and password:
                abort(str(e))

            # Otherwise, assume an auth exception, and prompt for new/better
            # password.
            #
            # Paramiko doesn't handle prompting for locked private keys (i.e.
            # keys with a passphrase and not loaded into an agent) so we have
            # to detect this and tweak our prompt slightly.  (Otherwise,
            # however, the logic flow is the same, because Paramiko's connect()
            # method overrides the password argument to be either the login
            # password OR the private key passphrase. Meh.)
            #
            # NOTE: This will come up if you normally use a
            # passphrase-protected private key with ssh-agent, and enter an
            # incorrect remote username, because Paramiko:
            #
            # * Tries the agent first, which will fail as you gave the wrong
            # username, so obviously any loaded keys aren't gonna work for a
            # nonexistent remote account;
            # * Then tries the on-disk key file, which is passphrased;
            # * Realizes there's no password to try unlocking that key with,
            # because you didn't enter a password, because you're using
            # ssh-agent;
            # * In this condition (trying a key file, password is None)
            # Paramiko raises PasswordRequiredException.
            #
            text = None
            if e.__class__ is ssh.PasswordRequiredException:
                # NOTE: we can't easily say WHICH key's passphrase is needed,
                # because Paramiko doesn't provide us with that info, and
                # env.key_filename may be a list of keys, so we can't know
                # which one raised the exception. Best not to try.
                prompt = "[%s] Passphrase for private key"
                text = prompt % env.host_string
            password = prompt_for_password(text, user=user)
            # Update env.password, env.passwords if empty
            set_password(password)
        # Ctrl-D / Ctrl-C for exit
        except (EOFError, TypeError):
            # Print a newline (in case user was sitting at prompt)
            print('')
            sys.exit(0)
        # Handle timeouts
        except timeout:
            abort('Timed out trying to connect to %s' % host)
        # Handle DNS error / name lookup failure
        except gaierror:
            abort('Name lookup failed for %s' % host)
        # Handle generic network-related errors
        # NOTE: In 2.6, socket.error subclasses IOError
        except socketerror, e:
            abort('Low level socket error connecting to host %s: %s' % (
                host, e[1])
            )

def prompt_for_password(prompt=None, no_colon=False, stream=None, user=None):
    """Prompt for and return a password."""

    stream = stream or sys.stderr
    if user:
        default = "[%s] Login password for user %s" % (env.host_string, user)
    else:
        default = "[%s] Login password" % env.host_string
    password_prompt = prompt if (prompt is not None) else default
    if not no_colon:
        password_prompt += ": "
    new_password = getpass(password_prompt, stream)
    attempts = 1
    while not new_password:
        attempts += 1
        if attempts > 3:
            abort("Too many login attempts.")
        new_password = getpass(password_prompt, stream)
    return new_password

def needs_host(func):
    """Prompt the user for the value of ``env.host_string`` if it's empty."""

    @wraps(func)
    def host_prompting_wrapper(*args, **kwargs):
        while not env.get('host_string', False):
            host_string = raw_input(
                "No hosts found. Please specify a host string for connection: "
                )
            interpret_host_string(host_string)
        return func(*args, **kwargs)
    return host_prompting_wrapper

def interpret_host_string(host_string):
    """Apply the given host string to the env dict."""

    username, hostname, port = normalize(host_string)
    env.host_string = host_string
    env.host = hostname
    env.user = username
    env.port = port
    return username, hostname, port

def disconnect_all():
    """Disconnect from all currently connected servers."""

    if env.colors:
        color = env.color_settings['finish']
    else:
        color = None
    for key in CONNECTIONS.keys():
        if output.status:
            msg = "Disconnecting from %s..." % denormalize(key)
            if color:
                msg = color(msg)
            print msg,
        CONNECTIONS[key].close()
        del CONNECTIONS[key]
        if output.status:
            if color:
                print color("done.")
            else:
                print "done."

# ------------------------------------------------------------------------------
# I/O Loops
# ------------------------------------------------------------------------------

def _flush(pipe, text):
    pipe.write(text)
    pipe.flush()

def _endswith(char_list, substring):
    tail = char_list[-1*len(substring):]
    substring = list(substring)
    return tail == substring

def output_loop(chan, which, capture):
    # Obtain stdout or stderr related values
    func = getattr(chan, which)
    if which == 'recv':
        prefix = "out"
        pipe = sys.stdout
    else:
        prefix = "err"
        pipe = sys.stderr
    # Allow prefix to be turned off.
    if env.output_prefix:
        host_prefix = "[%s]" % env.host_string
        if env.colors:
            host_prefix = env.color_settings['host_prefix'](host_prefix)
        prefix = "%s %s: " % (host_prefix, prefix)
    else:
        prefix = ""
    printing = getattr(output, 'stdout' if (which == 'recv') else 'stderr')
    # Initialize loop variables
    reprompt = False
    initial_prefix_printed = False
    while 1:
        # Handle actual read/write
        byte = func(1)
        if byte == '':
            break
        # A None capture variable implies that we're in open_shell()
        if capture is None:
            # Just print directly -- no prefixes, no capturing, nada
            # And since we know we're using a pty in this mode, just go
            # straight to stdout.
            _flush(sys.stdout, byte)
        # Otherwise, we're in run/sudo and need to handle capturing and
        # prompts.
        else:
            # Print to user
            if printing:
                # Initial prefix
                if not initial_prefix_printed:
                    _flush(pipe, prefix)
                    initial_prefix_printed = True
                # Byte itself
                _flush(pipe, byte)
                # Trailing prefix to start off next line
                if byte in ("\n", "\r"):
                    _flush(pipe, prefix)
            # Store in capture buffer
            capture += byte
            # Handle prompts
            prompt = _endswith(capture, env.sudo_prompt)
            try_again = (_endswith(capture, env.again_prompt + '\n')
                or _endswith(capture, env.again_prompt + '\r\n'))
            if prompt:
                # Obtain cached password, if any
                password = get_password()
                # Remove the prompt itself from the capture buffer. This is
                # backwards compatible with Fabric 0.9.x behavior; the user
                # will still see the prompt on their screen (no way to avoid
                # this) but at least it won't clutter up the captured text.
                del capture[-1*len(env.sudo_prompt):]
                # If the password we just tried was bad, prompt the user again.
                if (not password) or reprompt:
                    # Print the prompt and/or the "try again" notice if
                    # output is being hidden. In other words, since we need
                    # the user's input, they need to see why we're
                    # prompting them.
                    if not printing:
                        _flush(pipe, prefix)
                        if reprompt:
                            _flush(pipe, env.again_prompt + '\n' + prefix)
                        _flush(pipe, env.sudo_prompt)
                    # Prompt for, and store, password. Give empty prompt so the
                    # initial display "hides" just after the actually-displayed
                    # prompt from the remote end.
                    password = prompt_for_password(
                        prompt=" ", no_colon=True, stream=pipe
                    )
                    # Update env.password, env.passwords if necessary
                    set_password(password)
                    # Reset reprompt flag
                    reprompt = False
                # Send current password down the pipe
                chan.sendall(password + '\n')
            elif try_again:
                # Remove text from capture buffer
                capture = capture[:len(env.again_prompt)]
                # Set state so we re-prompt the user at the next prompt.
                reprompt = True

def input_loop(chan, using_pty):
    while not chan.exit_status_ready():
        if win32:
            have_char = msvcrt.kbhit()
        else:
            r, w, x = select([sys.stdin], [], [], 0.0)
            have_char = (r and r[0] == sys.stdin)
        if have_char:
            # Send all local stdin to remote end's stdin
            byte = msvcrt.getch() if win32 else sys.stdin.read(1)
            chan.sendall(byte)
            # Optionally echo locally, if needed.
            if not using_pty and env.echo_stdin:
                # Not using fastprint() here -- it prints as 'user'
                # output level, don't want it to be accidentally hidden
                sys.stdout.write(byte)
                sys.stdout.flush()
        sleep(IO_SLEEP)

# ------------------------------------------------------------------------------
# SFTP Support
# ------------------------------------------------------------------------------

class SFTP(object):
    """SFTP helper class, which is also a facade for paramiko.SFTPClient."""

    def __init__(self, host_string):
        self.ftp = CONNECTIONS[host_string].open_sftp()

    def __getattr__(self, attr):
        return getattr(self.ftp, attr)

    def isdir(self, path):
        try:
            return S_ISDIR(self.ftp.lstat(path).st_mode)
        except IOError:
            return False

    def islink(self, path):
        try:
            return S_ISLNK(self.ftp.lstat(path).st_mode)
        except IOError:
            return False

    def exists(self, path):
        try:
            self.ftp.lstat(path).st_mode
        except IOError:
            return False
        return True

    def glob(self, path):
        dirpart, pattern = split(path)
        rlist = self.ftp.listdir(dirpart)
        names = fnfilter([f for f in rlist if not f[0] == '.'], pattern)
        ret = [path]
        if len(names):
            s = '/'
            ret = [dirpart.rstrip(s) + s + name.lstrip(s) for name in names]
            if not win32:
                ret = [join(dirpart, name) for name in names]
        return ret

    def walk(self, top, topdown=True, onerror=None, followlinks=False):
        # We may not have read permission for top, in which case we can't get a
        # list of the files the directory contains. os.path.walk always
        # suppressed the exception then, rather than blow up for a minor reason
        # when (say) a thousand readable directories are still left to visit.
        # That logic is copied here.
        try:
            names = self.ftp.listdir(top)
        except Exception, err:
            if onerror is not None:
                onerror(err)
            return

        dirs, nondirs = [], []
        for name in names:
            if self.isdir(join(top, name)):
                dirs.append(name)
            else:
                nondirs.append(name)

        if topdown:
            yield top, dirs, nondirs

        for name in dirs:
            path = join(top, name)
            if followlinks or not self.islink(path):
                for x in self.walk(path, topdown, onerror, followlinks):
                    yield x

        if not topdown:
            yield top, dirs, nondirs

    def mkdir(self, path, use_sudo):
        if use_sudo:
            with hide('everything'):
                sudo('mkdir %s' % path)
        else:
            self.ftp.mkdir(path)

    def get(self, remote_path, local_path, local_is_path, rremote=None):
        # rremote => relative remote path, so get(/var/log) would result in
        # this function being called with
        # remote_path=/var/log/apache2/access.log and
        # rremote=apache2/access.log
        rremote = rremote if rremote is not None else remote_path
        # Handle format string interpolation (e.g. %(dirname)s)
        path_vars = {
            'host': env.host_string.replace(':', '-'),
            'basename': basename(rremote),
            'dirname': dirname(rremote),
            'path': rremote
        }
        if local_is_path:
            # Interpolate, then abspath (to make sure any /// are compressed)
            local_path = abspath(local_path % path_vars)
            # Ensure we give Paramiko a file by prepending and/or creating
            # local directories as appropriate.
            dirpath, filepath = split(local_path)
            if dirpath and not exists(dirpath):
                makedirs(dirpath)
            if isdir(local_path):
                local_path = join(local_path, path_vars['basename'])
        if output.running:
            print("[%s] download: %s <- %s" % (
                env.host_string,
                local_path if local_is_path else "<file obj>",
                remote_path
            ))
        # Warn about overwrites, but keep going
        if local_is_path and exists(local_path):
            msg = "Local file %s already exists and is being overwritten."
            warn(msg % local_path)
        # Have to bounce off FS if doing file-like objects
        fd, real_local_path = None, local_path
        if not local_is_path:
            fd, real_local_path = mkstemp()
        self.ftp.get(remote_path, real_local_path)
        # Return file contents (if it needs stuffing into a file-like obj)
        # or the final local file path (otherwise)
        result = None
        if not local_is_path:
            file_obj = fdopen(fd)
            result = file_obj.read()
            # Clean up temporary file
            file_obj.close()
            remove(real_local_path)
        else:
            result = real_local_path
        return result

    def get_dir(self, remote_path, local_path):

        # Decide what needs to be stripped from remote paths so they're all
        # relative to the given remote_path
        if basename(remote_path):
            strip = dirname(remote_path)
        else:
            strip = dirname(dirname(remote_path))

        # Store all paths gotten so we can return them when done
        result = []

        # Use our facsimile of os.walk to find all files within remote_path
        for context, dirs, files in self.walk(remote_path):

            # Normalize current directory to be relative
            # E.g. remote_path of /var/log and current dir of /var/log/apache2
            # would be turned into just 'apache2'
            lcontext = rcontext = context.replace(strip, '', 1).lstrip('/')

            # Prepend local path to that to arrive at the local mirrored
            # version of this directory. So if local_path was 'mylogs', we'd
            # end up with 'mylogs/apache2'
            lcontext = join(local_path, lcontext)

            # Download any files in current directory
            for f in files:
                # Construct full and relative remote paths to this file
                rpath = join(context, f)
                rremote = join(rcontext, f)
                # If local_path isn't using a format string that expands to
                # include its remote path, we need to add it here.
                if "%(path)s" not in local_path \
                    and "%(dirname)s" not in local_path:
                    lpath = join(lcontext, f)
                # Otherwise, just passthrough local_path to self.get()
                else:
                    lpath = local_path
                # Now we can make a call to self.get() with specific file paths
                # on both ends.
                result.append(self.get(rpath, lpath, True, rremote))

        return result

    def put(
        self, local_path, remote_path, use_sudo, mirror_local_mode, mode,
        local_is_path
        ):

        pre = self.ftp.getcwd()
        pre = pre if pre else ''

        if local_is_path and self.isdir(remote_path):
            remote_path = join(remote_path, basename(local_path))

        if output.running:
            print("[%s] put: %s -> %s" % (
                env.host_string,
                local_path if local_is_path else '<file obj>',
                join(pre, remote_path)
            ))

        # When using sudo, "bounce" the file through a guaranteed-unique file
        # path in the default remote CWD (which, typically, the login user will
        # have write permissions on) in order to sudo(mv) it later.
        if use_sudo:
            target_path = remote_path
            hasher = sha1()
            hasher.update(env.host_string)
            hasher.update(target_path)
            remote_path = hasher.hexdigest()

        # Have to bounce off FS if doing file-like objects
        fd, real_local_path = None, local_path
        if not local_is_path:
            fd, real_local_path = mkstemp()
            old_pointer = local_path.tell()
            local_path.seek(0)
            file_obj = fdopen(fd, 'wb')
            file_obj.write(local_path.read())
            file_obj.close()
            local_path.seek(old_pointer)

        rattrs = self.ftp.put(real_local_path, remote_path)
        # Clean up
        if not local_is_path:
            remove(real_local_path)

        # Handle modes if necessary
        if (local_is_path and mirror_local_mode) or (mode is not None):
            lmode = stat(local_path).st_mode if mirror_local_mode else mode
            lmode = lmode & 07777
            rmode = rattrs.st_mode & 07777
            if lmode != rmode:
                if use_sudo:
                    with hide('everything'):
                        sudo('chmod %o \"%s\"' % (lmode, remote_path))
                else:
                    self.ftp.chmod(remote_path, lmode)

        if use_sudo:
            with hide('everything'):
                sudo("mv \"%s\" \"%s\"" % (remote_path, target_path))
            # Revert to original remote_path for return value's sake
            remote_path = target_path

        return remote_path

    def put_dir(
        self, local_path, remote_path, use_sudo, mirror_local_mode, mode
        ):
        if basename(local_path):
            strip = dirname(local_path)
        else:
            strip = dirname(dirname(local_path))
        remote_paths = []
        for context, dirs, files in walk(local_path):
            rcontext = context.replace(strip, '', 1)
            rcontext = rcontext.lstrip('/')
            rcontext = join(remote_path, rcontext)
            if not self.exists(rcontext):
                self.mkdir(rcontext, use_sudo)
            for d in dirs:
                n = join(rcontext,d)
                if not self.exists(n):
                    self.mkdir(n, use_sudo)
            for f in files:
                local_path = join(context,f)
                n = join(rcontext,f)
                p = self.put(local_path, n, use_sudo, mirror_local_mode, mode,
                    True)
                remote_paths.append(p)
        return remote_paths

# ------------------------------------------------------------------------------
# Operations
# ------------------------------------------------------------------------------

class AttrString(str):
    """Simple string subclass to allow arbitrary attribute access."""

    @property
    def stdout(self):
        return str(self)

class AttrList(list):
    """Simple list subclass to allow arbitrary attribute access."""

def _pty_size():
    """Obtain (rows, cols) tuple for sizing a pty on the remote end."""

    rows, cols = 24, 80
    if not win32 and sys.stdin.isatty():
        fmt = 'HH'
        buffer = struct.pack(fmt, 0, 0)
        try:
            result = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ,
                buffer)
            rows, cols = struct.unpack(fmt, result)
        except AttributeError:
            pass
    return rows, cols

def _handle_failure(message, exception=None):
    """Call `abort` or `warn` with the given message."""

    func = env.warn_only and warn or abort
    if output.debug:
        message += "\n\n" + format_exc()
    elif exception is not None:
        if hasattr(exception, 'strerror') and exception.strerror is not None:
            underlying = exception.strerror
        else:
            underlying = exception
        message += "\n\nUnderlying exception message:\n" + indent(underlying)
    return func(message)

def _shell_escape(string):
    """Escape double quotes, backticks and dollar signs in given ``string``."""

    for char in ('"', '$', '`'):
        string = string.replace(char, '\%s' % char)
    return string

def prompt(text, key=None, default='', validate=None):
    """Prompt user with ``text`` and return the input (like ``raw_input``)."""

    if key:
        previous_value = env.get(key)

    default_str = ""
    if default != '':
        default_str = " [%s] " % str(default).strip()
    else:
        default_str = " "

    prompt_str = text.strip() + default_str
    if env.colors:
        prompt_str = env.color_settings['prompt'](prompt_str)

    value = None
    while value is None:
        value = raw_input(prompt_str) or default
        if validate:
            if callable(validate):
                try:
                    value = validate(value)
                except Exception, e:
                    value = None
                    msg = (
                        "Validation failed for the following reason:\n%s"
                        % indent(e.message)
                        )
                    if env.colors:
                        msg = env.color_settings['error'](msg)
                    print(msg)
            else:
                if not validate.startswith('^'):
                    validate = r'^' + validate
                if not validate.endswith('$'):
                    validate += r'$'
                result = findall(validate, value)
                if not result:
                    msg = (
                        "Regex validation failed: '%s' does not match '%s'"
                        % (value, validate)
                        )
                    if env.colors:
                        msg = env.color_settings['error'](msg)
                    print(msg)
                    value = None

    if key:
        env[key] = value

    if key and previous_value is not None and previous_value != value:
        warn("overwrote previous env variable '%s'; used to be '%s', is now '%s'." % (
            key, previous_value, value
        ))

    return value

@needs_host
def put(
    local_path=None, remote_path=None, use_sudo=False, mirror_local_mode=False,
    mode=None
    ):
    """Upload one or more files to a remote host."""

    ftp = SFTP(env.host_string)
    local_path = local_path or getcwd()
    local_is_path = not (hasattr(local_path, 'read')
                         and callable(local_path.read))

    with closing(ftp) as ftp:

        home = ftp.normalize('.')
        remote_path = remote_path or home
        if not isabs(remote_path) and env.get('cwd'):
            remote_path = env.cwd.rstrip('/') + '/' + remote_path

        if local_is_path:
            local_path = expanduser(local_path)
            if not isabs(local_path) and env.lcwd:
                local_path = join(env.lcwd, local_path)
            names = glob(local_path)
        else:
            names = [local_path]

        if local_is_path and not names:
            err = "'%s' is not a valid local path or glob." % local_path
            raise ValueError(err)

        if ftp.exists(remote_path):
            if local_is_path and len(names) != 1 and not ftp.isdir(remote_path):
                raise ValueError("'%s' is not a directory" % remote_path)

        remote_paths = []
        failed_local_paths = []
        for lpath in names:
            try:
                if local_is_path and isdir(lpath):
                    p = ftp.put_dir(lpath, remote_path, use_sudo,
                        mirror_local_mode, mode)
                    remote_paths.extend(p)
                else:
                    p = ftp.put(lpath, remote_path, use_sudo, mirror_local_mode,
                        mode, local_is_path)
                    remote_paths.append(p)
            except Exception, e:
                msg = "put() encountered an exception while uploading '%s'"
                failure = lpath if local_is_path else "<StringIO>"
                failed_local_paths.append(failure)
                _handle_failure(message=msg % lpath, exception=e)

        ret = AttrList(remote_paths)
        ret.failed = failed_local_paths
        ret.succeeded = not ret.failed
        return ret

@needs_host
def get(remote_path, local_path=None):
    """Download one or more files from a remote host."""

    ftp = SFTP(env.host_string)
    local_path = local_path or "%(host)s/%(path)s"
    local_is_path = not (hasattr(local_path, 'write')
                         and callable(local_path.write))

    if local_is_path and not isabs(local_path) and env.lcwd:
        local_path = join(env.lcwd, local_path)

    with closing(ftp) as ftp:
        home = ftp.normalize('.')
        if remote_path.startswith('~'):
            remote_path = remote_path.replace('~', home, 1)
        if local_is_path:
            local_path = expanduser(local_path)

        if not isabs(remote_path):
            if env.get('cwd'):
                remote_path = env.cwd.rstrip('/') + '/' + remote_path
            else:
                remote_path = join(home, remote_path)

        local_files = []
        failed_remote_files = []

        try:
            names = ftp.glob(remote_path)

            if not local_is_path:
                if len(names) > 1 or ftp.isdir(names[0]):
                    _handle_failure("[%s] %s is a glob or directory, but local_path is a file object!" % (env.host_string, remote_path))

            for remote_path in names:
                if ftp.isdir(remote_path):
                    result = ftp.get_dir(remote_path, local_path)
                    local_files.extend(result)
                else:
                    result = ftp.get(
                        remote_path, local_path, local_is_path,
                        basename(remote_path)
                        )
                    if not local_is_path:
                        local_path.seek(0)
                        local_path.write(result)
                    else:
                        local_files.append(result)

        except Exception, e:
            failed_remote_files.append(remote_path)
            msg = "get() encountered an exception while downloading '%s'"
            _handle_failure(message=msg % remote_path, exception=e)

        ret = AttrList(local_files if local_is_path else [])
        ret.failed = failed_remote_files
        ret.succeeded = not ret.failed
        return ret

def _sudo_prefix(user):
    """Return ``env.sudo_prefix`` with ``user`` inserted if necessary."""

    prefix = env.sudo_prefix % env.sudo_prompt
    if user is not None:
        if str(user).isdigit():
            user = "#%s" % user
        return "%s -u \"%s\" " % (prefix, user)
    return prefix

def _shell_wrap(command, shell=True, sudo_prefix=None):
    """Conditionally wrap given command in env.shell (while honoring sudo.)"""

    if shell and not env.use_shell:
        shell = False
    if sudo_prefix is None:
        sudo_prefix = ""
    else:
        sudo_prefix += " "
    if shell:
        shell = env.shell + " "
        command = '"%s"' % _shell_escape(command)
    else:
        shell = ""
    return sudo_prefix + shell + command

def _prefix_commands(command, which, dir=None):
    """Prefixes ``command`` with ``env.command_prefixes``."""

    prefixes = list(env.command_prefixes)
    if dir:
        dir = dir.replace(' ', r'\ ')
        prefixes.append('cd %s' % dir)
    cwd = env.cwd if which == 'remote' else env.lcwd
    if cwd:
        prefixes.insert(0, 'cd %s' % cwd)
    if prefixes:
        return " && ".join(prefixes) + " && " + command
    return command

def _prefix_env_vars(command):
    """Prefixes ``command`` with shell environment vars, e.g. ``PATH=foo ``."""

    env_vars = [
        'export %s' % stringify_env_var(key[1:])
        for key in env if key.startswith('$')
        ]
    if env_vars:
        return ' && '.join(env_vars) + ' && ' + command
    return command

def _execute(
    channel, command, pty=True, combine_stderr=True, invoke_shell=False
    ):
    """Execute ``command`` over ``channel``."""

    with char_buffered(sys.stdin):

        if combine_stderr or env.combine_stderr:
            channel.set_combine_stderr(True)

        using_pty = True
        if not invoke_shell and (not pty or not env.always_use_pty):
            using_pty = False

        if using_pty:
            rows, cols = _pty_size()
            channel.get_pty(width=cols, height=rows)

        if invoke_shell:
            channel.invoke_shell()
            if command:
                channel.sendall(command + "\n")
        else:
            channel.exec_command(command)

        stdout, stderr = [], []
        if invoke_shell:
            stdout = stderr = None

        workers = (
            ThreadHandler('out', output_loop, channel, "recv", stdout),
            ThreadHandler('err', output_loop, channel, "recv_stderr", stderr),
            ThreadHandler('in', input_loop, channel, using_pty)
        )

        while 1:
            if channel.exit_status_ready():
                break
            else:
                for worker in workers:
                    e = worker.exception
                    if e:
                        raise e[0], e[1], e[2]
            sleep(IO_SLEEP)

        status = channel.recv_exit_status()
        for worker in workers:
            worker.thread.join()

        channel.close()
        if not invoke_shell:
            stdout = ''.join(stdout).strip()
            stderr = ''.join(stderr).strip()

        if output.running \
            and (output.stdout and stdout and not stdout.endswith("\n")) \
            or (output.stderr and stderr and not stderr.endswith("\n")):
            print("")

        return stdout, stderr, status

@needs_host
def open_shell(command=None):
    """Invoke a fully interactive shell on the remote end."""

    _execute(default_channel(), command, True, True, True)

def _run_command(
    command, shell=True, pty=True, combine_stderr=True, sudo=False, user=None,
    dir=None, format=True
    ):

    if format:
        command = command.format(**env)

    given_command = command
    wrapped_command = _shell_wrap(
        _prefix_env_vars(_prefix_commands(command, 'remote', dir)),
        shell,
        _sudo_prefix(user) if sudo else None
        )

    which = 'sudo' if sudo else 'run'
    prefix = "[%s]" % env.host_string
    if env.colors:
        prefix = env.color_settings['host_prefix'](prefix)

    if output.debug:
        print("%s %s: %s" % (prefix, which, wrapped_command))
    elif output.running:
        print("%s %s: %s" % (prefix, which, given_command))

    stdout, stderr, status = _execute(
        default_channel(), wrapped_command, pty, combine_stderr
        )

    out = AttrString(stdout)
    err = AttrString(stderr)

    out.failed = False
    if status != 0:
        out.failed = True
        msg = "%s() encountered an error (return code %s) while executing '%s'" % (which, status, command)
        _handle_failure(message=msg)

    out.return_code = status
    out.succeeded = not out.failed
    out.stderr = err
    return out

@needs_host
def run(
    command, shell=True, pty=True, combine_stderr=True, dir=None, format=True
    ):
    """Run a shell command on a remote host."""

    return _run_command(
        command, shell, pty, combine_stderr, dir=dir, format=format
        )

@needs_host
def sudo(
    command, shell=True, pty=True, combine_stderr=True, user=None, dir=None,
    format=True
    ):
    """Run a shell command on a remote host, with superuser privileges."""

    return _run_command(
        command, shell, pty, combine_stderr, True, user, dir, format
        )

def local(command, capture=False, dir=None, format=True):
    """Run a command on the local system."""

    if format:
        command = command.format(**env)

    given_command = command
    wrapped_command = _prefix_env_vars(_prefix_commands(command, 'local', dir))
    prefix = "[localhost]"
    if env.colors:
        prefix = env.color_settings['host_prefix'](prefix)

    if output.debug:
        print("%s local: %s" % (prefix, wrapped_command))
    elif output.running:
        print("%s local: %s" % (prefix, given_command))

    dev_null = None
    if capture:
        out_stream = PIPE
        err_stream = PIPE
    else:
        dev_null = open(devnull, 'w+')
        out_stream = None if output.stdout else dev_null
        err_stream = None if output.stderr else dev_null
    try:
        cmd_arg = wrapped_command if win32 else [wrapped_command]
        p = Popen(cmd_arg, shell=True, stdout=out_stream, stderr=err_stream)
        (stdout, stderr) = p.communicate()
    finally:
        if dev_null is not None:
            dev_null.close()

    out = AttrString(stdout.strip() if stdout else "")
    err = AttrString(stderr.strip() if stderr else "")
    out.failed = False
    out.return_code = p.returncode
    out.stderr = err

    if p.returncode != 0:
        out.failed = True
        msg = "local() encountered an error (return code %s) while executing '%s'" % (p.returncode, command)
        _handle_failure(message=msg)

    out.succeeded = not out.failed
    return out

@needs_host
def reboot(wait):
    """Reboot the remote system, disconnect, and wait for ``wait`` seconds."""

    sudo('reboot')
    client = CONNECTIONS[env.host_string]
    client.close()
    if env.host_string in CONNECTIONS:
        del CONNECTIONS[env.host_string]
    if output.running:
        fastprint("Waiting for reboot: ", show_host=1)
        per_tick = 5
        for second in range(int(wait / per_tick)):
            fastprint(".")
            sleep(per_tick)
        fastprint("done.\n")

def confirm(question, default_for_empty_response=True):
    """Return the response to a yes/no question."""

    if default_for_empty_response:
        suffix = "Y/n"
    else:
        suffix = "y/N"

    while 1:
        response = prompt("%s [%s] " % (question, suffix)).lower()
        if not response:
            return default_for_empty_response
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("Please specify '(y)es' or '(n)o'.")
