# Public Domain (-) 2011 The Bolt Authors.
# See the Bolt UNLICENSE file for details.

"""Bolt Core."""

import atexit
import sys

from cPickle import dumps, loads
from fnmatch import fnmatch
from imp import load_source
from math import ceil
from optparse import OptionParser
from os import X_OK, access, chdir, environ, getcwd, listdir, pathsep, pipe
from os.path import abspath, dirname, exists, expanduser, isabs, isdir, join
from os.path import normpath, realpath, sep
from random import sample
from socket import AF_UNIX, SOCK_STREAM, error as socketerror, socket
from struct import calcsize, pack, unpack
from textwrap import dedent
from time import time
from traceback import format_exc
from uuid import uuid4

from Crypto.Random import atfork
from tavutil.io import DEVNULL
from tavutil.optcomplete import ListCompleter, autocomplete
from yaml import safe_load as load_yaml

from bolt.fabcode import (
    HOST_REGEX, abort, AttrDict, blue, cyan, disconnect_all, _escape_split, env,
    fastprint, _get_system_username, green, hide, indent, local, output, puts,
    _rc_path, reboot, red, run, _setenv, settings, stringify_env_var, show,
    sudo, yellow, warn
    )

try:
    from errno import EAGAIN, EINTR, EPIPE
    from os import close, fork, kill, read, remove, write
    from signal import SIGALRM, SIGTERM, alarm, signal
except ImportError:
    forkable = False
else:
    forkable = True

# ------------------------------------------------------------------------------
# Some Constants
# ------------------------------------------------------------------------------

__version__ = "0.9"

INDEX_HEADER_SIZE = calcsize('H')
SHELL_BUILTINS = {}
SHELL_HISTORY_FILE = None

CACHE = {}
DEFAULT = {}
HOSTINFO = {}
HOSTPATTERNINFO = {}
HOSTPATTERNS = []

# ------------------------------------------------------------------------------
# Default Settings Loader
# ------------------------------------------------------------------------------

def get_settings(
    contexts, env=env, cache=CACHE, default=DEFAULT, hostinfo=HOSTINFO,
    hostpatterninfo=HOSTPATTERNINFO, hostpatterns=HOSTPATTERNS
    ):
    """Return a sequence of host/settings for the given contexts tuple."""

    # Exit early for null contexts.
    if not contexts:
        return []

    # Check the cache.
    if contexts in cache:
        return cache[contexts]

    # Mimick @hosts-like behaviour when there's no env.config.
    if 'config' not in env:
        responses = []; out = responses.append
        for host in contexts:
            if host and (('.' in host) or (host == 'localhost')):
                resp = {'host_string': host}
                info = HOST_REGEX.match(host).groupdict()
                resp['host'] = info['host']
                resp['port'] = info['port'] or '22'
                resp['user'] = info['user'] or env.get('user')
                out(resp)
        return cache.setdefault(contexts, responses)

    # Save env.config to a local parameter to avoid repeated lookup.
    config = env.config

    # Set a marker to handle the first time.
    if not cache:

        cache['_init'] = 1

        # Grab the root default settings.
        if 'default' in config:
            default.update(config.default)

        # Grab any host specific settings.
        if 'hostinfo' in config:
            for host, info in config.hostinfo.items():
                if ('*' in host) or ('?' in host) or ('[' in host):
                    hostpatterninfo[host] = info
                else:
                    hostinfo[host] = info
            if hostpatterninfo:
                hostpatterns[:] = sorted(hostpatterninfo)

        def get_host_info(context, init=None):
            resp = default.copy()
            if init:
                resp.update(init)
            info = HOST_REGEX.match(context).groupdict()
            host = info['host']
            for pattern in hostpatterns:
                if fnmatch(host, pattern):
                    resp.update(hostpatterninfo[pattern])
                if fnmatch(context, pattern):
                    resp.update(hostpatterninfo[pattern])
            if host in hostinfo:
                resp.update(hostinfo[host])
            if context in hostinfo:
                resp.update(hostinfo[context])
            resp['host'] = host
            resp['host_string'] = context
            if info['port']:
                resp['port'] = info['port']
            elif 'port' not in resp:
                resp['port'] = '22'
            if info['user']:
                resp['user'] = info['user']
            elif 'user' not in resp:
                resp['user'] = env.user
            return resp

        get_settings.get_host_info = get_host_info

    else:
        get_host_info = get_settings.get_host_info

    # Loop through the contexts gathering host/settings.
    responses = []; out = responses.append
    for context in contexts:

        # Handle composite contexts.
        if '/' in context:
            context, hosts = context.split('/', 1)
            hosts = hosts.split(',')
            base = config[context].copy()
            additional = {}
            for _host in base.pop('hosts', []):
                if isinstance(_host, dict):
                    _host, _additional = _host.items()[0]
                    additional[_host] = _additional
            for host in hosts:
                if host in additional:
                    resp = get_host_info(host, base)
                    resp.update(additional[host])
                    out(resp)
                else:
                    out(get_host_info(host, base))

        # Handle hosts.
        elif ('.' in context) or (context == 'localhost'):
            out(get_host_info(context))

        else:
            base = config[context].copy()
            hosts = base.pop('hosts')
            for host in hosts:
                if isinstance(host, basestring):
                    out(get_host_info(host, base))
                else:
                    if len(host) > 1:
                        raise ValueError(
                            "More than 1 host found in config:\n\n%r\n"
                            % host.items()
                            )
                    host, additional = host.items()[0]
                    resp = get_host_info(host, base)
                    resp.update(additional)
                    out(resp)

    return cache.setdefault(contexts, responses)

# ------------------------------------------------------------------------------
# Auto Environment Variables Support
# ------------------------------------------------------------------------------

class EnvManager(object):
    """Generator for environment variables-related context managers."""

    cache = {}

    def __init__(self, var):
        self.var = var

    @classmethod
    def for_var(klass, var):
        cache = klass.cache
        if var in cache:
            return cache[var]
        return cache.setdefault(var, klass(var))

    def __str__(self):
        return stringify_env_var(self.var)

    def __call__(
        self, value=None, behaviour='append', sep=pathsep, reset=False,
        _valid=frozenset(['append', 'prepend', 'replace'])
        ):
        if value is None:
            return stringify_env_var(self.var)
        if behaviour not in _valid:
            raise ValueError("Unknown behaviour: %s" % behaviour)
        key = '$%s' % self.var
        val = []
        if (not reset) and (behaviour != 'replace'):
            if key in env:
                val.extend(env[key])
        val.append((value, behaviour, sep))
        kwargs = {key: tuple(val)}
        return _setenv(**kwargs)

# ------------------------------------------------------------------------------
# Hooks Support
# ------------------------------------------------------------------------------

HOOKS = {}
DISABLED_HOOKS = []
ENABLED_HOOKS = []

def hook(*names):
    def register(func):
        for name in names:
            name = name.replace('_', '-')
            if name not in HOOKS:
                HOOKS[name] = []
            HOOKS[name].append(func)
        return func
    return register

def get_hooks(name, disabled=False):
    name = name.replace('_', '-')
    for pattern in DISABLED_HOOKS:
        if fnmatch(name, pattern):
            disabled = 1
    for pattern in ENABLED_HOOKS:
        if fnmatch(name, pattern):
            disabled = 0
    if disabled:
        return []
    return HOOKS.get(name, [])

def call_hooks(name, *args, **kwargs):
    name = name.replace('_', '-')
    prev_hook = env.hook
    env.hook = name
    try:
        for hook in get_hooks(name):
            hook(*args, **kwargs)
    finally:
        env.hook = prev_hook

hook.get = get_hooks
hook.call = call_hooks
hook.registry = HOOKS

# ------------------------------------------------------------------------------
# Timeout
# ------------------------------------------------------------------------------

class ProcessTimeout(object):
    """Process timeout indicator."""

    failed = 1
    succeeded = 0

    def __bool__(self):
        return False

    __nonzero__ = __bool__

    def __str__(self):
        return 'TIMEOUT'

    __repr__ = __str__

TIMEOUT = ProcessTimeout()

class TimeoutException(Exception):
    """An internal timeout exception raised on SIGALRM."""

# ------------------------------------------------------------------------------
# Proxy Boolean
# ------------------------------------------------------------------------------

class WarningBoolean(object):
    """Proxy boolean to env.warning."""

    def __bool__(self):
        return env.warn_only

    __nonzero__ = __bool__


WarnOnly = WarningBoolean()

# ------------------------------------------------------------------------------
# Failure Handler
# ------------------------------------------------------------------------------

def handle_failure(cmd, warn_only):
    if hasattr(cmd, '__name__'):
        cmd = cmd.__name__ + '()'
    message = 'Error running `%s`\n\n%s' % (cmd, indent(format_exc()))
    if warn_only:
        warn(message)
    else:
        abort(message)

# ------------------------------------------------------------------------------
# Shell Spec
# ------------------------------------------------------------------------------

class ShellSpec(object):
    """Container class for shell spec variables."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# ------------------------------------------------------------------------------
# Response List
# ------------------------------------------------------------------------------

class ResponseList(list):
    """Container class for response values."""

    @classmethod
    def new(klass, settings, value=None):
        if value:
            obj = klass(value)
        else:
            obj = klass()
        obj._settings = settings
        return obj

    def ziphost(self):
        for response, setting in zip(self, self._settings):
            yield response, setting['host_string']

    def zipsetting(self):
        for response, setting in zip(self, self._settings):
            yield response, setting

    @property
    def settings(self):
        return self._settings[:]

# ------------------------------------------------------------------------------
# Execute Operation
# ------------------------------------------------------------------------------

DEFAULT_SCRIPT_NAME = 'fab.%s' % uuid4()

def execute(
    script, name=None, verbose=True, shell=True, pty=True, combine_stderr=True,
    dir=None
    ):
    """Run arbitrary scripts on a remote host."""

    script = dedent(script).strip()
    if verbose:
        prefix = "[%s]" % env.host_string
        if env.colors:
            prefix = env.color_settings['host_prefix'](prefix)
        print("%s run: %s" % (prefix, name or script))
    name = name or DEFAULT_SCRIPT_NAME
    with hide('running', 'stdout', 'stderr'):
        run('cat > ' + name + ' << FABEND\n' + script + '\nFABEND\n', dir=dir)
        run('chmod +x ' + name, dir=dir)
        try:
            if verbose > 1:
                with show('stdout', 'stderr'):
                    out = run('./' + name, shell, pty, combine_stderr, dir)
            else:
                out = run('./' + name, shell, pty, combine_stderr, dir)
        finally:
            run('rm ' + name, dir=dir)
    return out

# ------------------------------------------------------------------------------
# Core Context Class
# ------------------------------------------------------------------------------

class ContextRunner(object):
    """A convenience class to support operations on initialised contexts."""

    def __init__(self, *args, **kwargs):
        if kwargs and 'settings' in kwargs:
            self.ctx = ('<sample>',)
            self._settings = kwargs['settings']
        if args:
            if len(args) == 1 and not isinstance(args[0], basestring):
                args = tuple(args[0])
            self.ctx = args
            self._settings = env.get_settings(args)
        else:
            if env.ctx:
                self.ctx = env.ctx
                self._settings = env.get_settings(env.ctx)
            else:
                self.ctx = ()
                self._settings = []

    def execute(
        self, script, name=None, verbose=True, shell=True, pty=True,
        combine_stderr=True, dir=None
        ):
        ctx, e = self.ctx, execute
        settings_list = self._settings
        responses = ResponseList.new(settings_list); out = responses.append
        for kwargs in settings_list:
            with settings(ctx=ctx, **kwargs):
                out(e(script, name, verbose, shell, pty, combine_stderr, dir))
        return responses

    def local(self, command, capture=True, dir=None, format=True):
        ctx, l = self.ctx, local
        settings_list = self._settings
        responses = ResponseList.new(settings_list); out = responses.append
        for kwargs in settings_list:
            with settings(ctx=ctx, **kwargs):
                out(l(command, capture, dir, format))
        return responses

    def reboot(self, wait):
        ctx, r = self.ctx, reboot
        settings_list = self._settings
        responses = ResponseList.new(settings_list); out = responses.append
        for kwargs in settings_list:
            with settings(ctx=ctx, **kwargs):
                out(r(wait))
        return responses

    def run(
        self, command, shell=True, pty=True, combine_stderr=True, dir=None,
        format=True, warn_only=WarnOnly
        ):
        ctx = self.ctx
        settings_list = self._settings
        responses = ResponseList.new(settings_list); out = responses.append
        if isinstance(command, basestring):
            r = run
            for kwargs in settings_list:
                with settings(ctx=ctx, warn_only=warn_only, **kwargs):
                    out(r(command, shell, pty, combine_stderr, dir, format))
        else:
            for kwargs in settings_list:
                with settings(ctx=ctx, warn_only=warn_only, **kwargs):
                    try:
                        out(command())
                    except Exception, error:
                        out(error)
                        handle_failure(command, warn_only)
        return responses

    def shell(
        self, builtins=SHELL_BUILTINS, shell=True, pty=True,
        combine_stderr=True, dir=None, format=True, warn_only=True
        ):
        ctx = self.ctx
        settings_list = self._settings
        if not settings_list:
            return
        global SHELL_HISTORY_FILE
        if (not SHELL_HISTORY_FILE) and readline:
            SHELL_HISTORY_FILE = expanduser(env.shell_history_file)
            try:
                readline.read_history_file(SHELL_HISTORY_FILE)
            except IOError:
                pass
            atexit.register(readline.write_history_file, SHELL_HISTORY_FILE)
        fastprint("shell mode\n\n", 'system')
        spec = ShellSpec(
            shell=shell, pty=pty, combine_stderr=combine_stderr, dir=dir,
            format=format
            )
        r = run
        count = 0
        prefix = '>> '
        if env.colors:
            prefix = env.color_settings['prefix'](prefix)
        try:
            while 1:
                try:
                    command = raw_input(prefix).strip()
                except EOFError:
                    raise KeyboardInterrupt
                if not command:
                    continue
                builtin_cmd = 0
                if command.startswith('.'):
                    if (len(command) > 1) and command[1].isalpha():
                        builtin_cmd = 1
                if builtin_cmd:
                    command = command.split(' ', 1)
                    if len(command) == 1:
                        command = command[0]
                        arg = ''
                    else:
                        command, arg = command
                    command = command[1:].strip()
                    if not command:
                        continue
                    command = command.replace('_', '-')
                    if command not in builtins:
                        warn("Couldn't find builtin command %r" % command)
                        continue
                    command = builtins[command]
                    if hasattr(command, '__single__'):
                        with settings(ctx=ctx, warn_only=warn_only):
                            try:
                                command(spec, arg)
                            except Exception:
                                handle_failure(command, warn_only)
                        continue
                for kwargs in settings_list:
                    with settings(ctx=ctx, warn_only=warn_only, **kwargs):
                        try:
                            if builtin_cmd:
                                try:
                                    command(spec, arg)
                                except Exception:
                                    handle_failure(command, warn_only)
                            else:
                                r(command, spec.shell, spec.pty,
                                  spec.combine_stderr, spec.dir, spec.format)
                        except KeyboardInterrupt:
                            print
                            count += 1
                            if count > 2:
                                raise KeyboardInterrupt
                    count = 0
        except KeyboardInterrupt:
            print
            print
            fastprint("shell mode terminated\n", 'system')

    def sudo(
        self, command, shell=True, pty=True, combine_stderr=True, user=None,
        dir=None, format=True
        ):
        ctx, s = self.ctx, sudo
        settings_list = self._settings
        responses = ResponseList.new(settings_list); out = responses.append
        for kwargs in settings_list:
            with settings(ctx=ctx, **kwargs):
                out(s(command, shell, pty, combine_stderr, user, dir, format))
        return responses

    if forkable:

        def multilocal(
            self, command, capture=True, dir=None, format=True, warn_only=True,
            condensed=False, quiet_exit=True, laggards_timeout=None,
            wait_for=None
            ):
            def run_local():
                return local(command, capture, dir, format)
            return self.multirun(
                run_local, warn_only=warn_only, condensed=condensed,
                quiet_exit=quiet_exit, laggards_timeout=laggards_timeout,
                wait_for=wait_for
                )

        def multisudo(
            self, command, shell=True, pty=True, combine_stderr=True, user=None,
            dir=None, format=True, warn_only=True, condensed=False,
            quiet_exit=True, laggards_timeout=None, wait_for=None
            ):
            def run_sudo():
                return sudo(
                    command, shell, pty, combine_stderr, user, dir, format
                    )
            return self.multirun(
                run_sudo, warn_only=warn_only, condensed=condensed,
                quiet_exit=quiet_exit, laggards_timeout=laggards_timeout,
                wait_for=wait_for
                )

        def multirun(
            self, command, shell=True, pty=True, combine_stderr=True, dir=None,
            format=True, warn_only=True, condensed=False, quiet_exit=True,
            laggards_timeout=None, wait_for=None
            ):
            settings_list = self._settings
            if not settings_list:
                return ResponseList.new(settings_list)
            if laggards_timeout:
                if not isinstance(laggards_timeout, int):
                    raise ValueError(
                        "The laggards_timeout parameter must be an int."
                        )
                if isinstance(wait_for, float):
                    if not 0.0 <= wait_for <= 1.0:
                        raise ValueError(
                            "A float wait_for needs to be between 0.0 and 1.0"
                            )
                    wait_for = int(ceil(wait_for * len(settings_list)))
            env.disable_char_buffering = 1
            try:
                return self._multirun(
                    command, settings_list, shell, pty, combine_stderr, dir,
                    format, warn_only, condensed, quiet_exit, laggards_timeout,
                    wait_for
                    )
            finally:
                env.disable_char_buffering = 0

        def _multirun(
            self, command, settings_list, shell, pty, combine_stderr, dir,
            format, warn_only, condensed, quiet_exit, laggards_timeout,
            wait_for
            ):

            callable_command = hasattr(command, '__call__')
            done = 0
            idx = 0
            ctx = self.ctx
            processes = {}
            total = len(settings_list)
            pool_size = env.multirun_pool_size
            socket_path = '/tmp/fab.%s' % uuid4()

            server = socket(AF_UNIX, SOCK_STREAM)
            server.bind(socket_path)
            server.listen(pool_size)

            for client_id in range(min(pool_size, total)):
                from_parent, to_child = pipe()
                pid = fork()
                if pid:
                    processes[client_id] = [from_parent, to_child, pid, idx]
                    idx += 1
                    write(to_child, pack('H', idx))
                else:
                    atfork()
                    def die(*args):
                        if quiet_exit:
                            output.status = False
                            sys.exit()
                    signal(SIGALRM, die)
                    if condensed:
                        sys.__ori_stdout__ = sys.stdout
                        sys.__ori_stderr__ = sys.stderr
                        sys.stdout = sys.stderr = DEVNULL
                    while 1:
                        alarm(env.multirun_child_timeout)
                        data = read(from_parent, INDEX_HEADER_SIZE)
                        alarm(0)
                        idx = unpack('H', data)[0] - 1
                        if idx == -1:
                            die()
                        try:
                            if callable_command:
                                with settings(
                                    ctx=ctx, warn_only=warn_only,
                                    **settings_list[idx]
                                    ):
                                    try:
                                        response = command()
                                    except Exception, error:
                                        handle_failure(command, warn_only)
                                        response = error
                            else:
                                with settings(
                                    ctx=ctx, warn_only=warn_only,
                                    **settings_list[idx]
                                    ):
                                    response = run(
                                        command, shell, pty, combine_stderr,
                                        dir, format
                                        )
                        except BaseException, error:
                            response = error
                        client = socket(AF_UNIX, SOCK_STREAM)
                        client.connect(socket_path)
                        client.send(dumps((client_id, idx, response)))
                        client.close()


            if laggards_timeout:
                break_early = 0
                responses = [TIMEOUT] * total
                def timeout_handler(*args):
                    raise TimeoutException
                original_alarm_handler = signal(SIGALRM, timeout_handler)
                total_waited = 0.0
            else:
                responses = [None] * total

            if condensed:
                prefix = '[multirun]'
                if env.colors:
                    prefix = env.color_settings['prefix'](prefix)
                stdout = sys.stdout
                if callable_command:
                    command = '%s()' % command.__name__
                else:
                    command = command
                if total < pool_size:
                    print (
                        "%s Running %r on %s hosts" % (prefix, command, total)
                        )
                else:
                    print (
                        "%s Running %r on %s hosts with pool of %s" %
                        (prefix, command, total, pool_size)
                        )
                template = "%s %%s/%s completed ..." % (prefix, total)
                info = template % 0
                written = len(info) + 1
                stdout.write(info)
                stdout.flush()

            while done < total:
                if laggards_timeout:
                    try:
                        if wait_for and done >= wait_for:
                            wait_start = time()
                        alarm(laggards_timeout)
                        conn, addr = server.accept()
                    except TimeoutException:
                        if not wait_for:
                            break_early= 1
                            break
                        if done >= wait_for:
                            break_early = 1
                            break
                        continue
                    else:
                        alarm(0)
                        if wait_for and done >= wait_for:
                            total_waited += time() - wait_start
                            if total_waited > laggards_timeout:
                                break_early = 1
                                break
                else:
                    conn, addr = server.accept()
                stream = []; buffer = stream.append
                while 1:
                    try:
                        data = conn.recv(1024)
                    except socketerror, errmsg:
                        if errmsg.errno in [EAGAIN, EPIPE, EINTR]:
                            continue
                        raise
                    if not data:
                        break
                    buffer(data)
                client_id, resp_idx, response = loads(''.join(stream))
                responses[resp_idx] = response
                done += 1
                spec = processes[client_id]
                if idx < total:
                    spec[3] = idx
                    idx += 1
                    write(spec[1], pack('H', idx))
                else:
                    spec = processes.pop(client_id)
                    write(spec[1], pack('H', 0))
                    close(spec[0])
                    close(spec[1])
                if condensed:
                    stdout.write('\x08' * written)
                    print (
                        "%s Finished on %s" %
                        (prefix, settings_list[resp_idx]['host_string'])
                        )
                    if done == total:
                        info = "%s %s/%s completed successfully!" % (
                            prefix, done, done
                            )
                    else:
                        info = template % done
                    written = len(info) + 1
                    stdout.write(info)
                    stdout.flush()

            if laggards_timeout:
                if break_early:
                    for spec in processes.itervalues():
                        kill(spec[2], SIGTERM)
                    if condensed:
                        stdout.write('\x08' * written)
                        info = "%s %s/%s completed ... laggards discarded!" % (
                            prefix, done, total
                            )
                        stdout.write(info)
                        stdout.flush()
                signal(SIGALRM, original_alarm_handler)

            if condensed:
                stdout.write('\n')
                stdout.flush()

            server.close()
            remove(socket_path)

            return ResponseList.new(settings_list, responses)

    else:

        def multilocal(self, *args, **kwargs):
            abort("multilocal is not supported on this setup")

        def multirun(self, *args, **kwargs):
            abort("multirun is not supported on this setup")

        def multisudo(self, *args, **kwargs):
            abort("multisudo is not supported on this setup")

    def select(self, filter):
        if isinstance(filter, int):
            return ContextRunner(settings=sample(self._settings, filter))
        return ContextRunner(settings=filter(self._settings[:]))

    @property
    def settings(self):
        return self._settings[:]

# ------------------------------------------------------------------------------
# Utility API Functions
# ------------------------------------------------------------------------------

def failed(responses):
    """Utility function that returns True if any of the responses failed."""

    return any(isinstance(resp, Exception) or resp.failed for resp in responses)

def succeeded(responses):
    """Utility function that returns True if the responses all succeeded."""

    return all(
        (not isinstance(resp, Exception)) and resp.succeeded
        for resp in responses
        )

def shell(name_or_func=None, single=False):
    """Decorator to register shell builtin commands."""

    if name_or_func:
        if isinstance(name_or_func, basestring):
            name = name_or_func
            func = None
        else:
            name = name_or_func.__name__
            func = name_or_func
    else:
        name = func = None
    if func:
        SHELL_BUILTINS[name.replace('_', '-')] = func
        if single:
            func.__single__ = 1
        return func
    def __decorate(func):
        SHELL_BUILTINS[(name or func.__name__).replace('_', '-')] = func
        if single:
            func.__single__ = 1
        return func
    return __decorate

# ------------------------------------------------------------------------------
# Default Shell Builtins
# ------------------------------------------------------------------------------

@shell(single=True)
def info(spec, arg):
    """list the hosts and the current context"""

    print
    print "Context:"
    print
    print "\n".join("   %s" % ctx for ctx in env.ctx)
    print
    print "Hosts:"
    print
    for setting in env().settings:
        print "  ", setting['host_string']
    print

@shell(single=True)
def cd(spec, arg):
    """change to a new working directory"""

    arg = arg.strip()
    if arg:
        if isabs(arg):
            spec.dir = arg
        elif arg.startswith('~'):
            spec.dir = expanduser(arg)
        else:
            if spec.dir:
                spec.dir = join(spec.dir, arg)
            else:
                spec.dir = join(getcwd(), arg)
        spec.dir = normpath(spec.dir)
        print "Switched to:", spec.dir
    else:
        spec.dir = None

@shell('local', single=True)
def builtin_local(spec, arg):
    """run the command locally"""

    local(arg, capture=0, dir=spec.dir, format=spec.format)

@shell('sudo')
def builtin_sudo(spec, arg):
    """run the sudoed command on remote hosts"""

    return sudo(
        arg, spec.shell, spec.pty, spec.combine_stderr, None, spec.dir,
        spec.format
        )

@shell(single=True)
def toggle_format(spec, arg):
    """toggle string formatting support"""

    format = spec.format
    if format:
        spec.format = False
        print "Formatting disabled."
    else:
        spec.format = True
        print "Formatting enabled."

@shell(single=True)
def multilocal(spec, arg):
    """run the command in parallel locally for each host"""

    def run_local():
        return local(arg, capture=0, dir=spec.dir, format=spec.format)

    env().multirun(
        run_local, spec.shell, spec.pty, spec.combine_stderr, spec.dir,
        spec.format, quiet_exit=1
        )

@shell(single=True)
def multirun(spec, arg):
    """run the command in parallel on the various hosts"""

    env().multirun(
        arg, spec.shell, spec.pty, spec.combine_stderr, spec.dir, spec.format,
        quiet_exit=1
        )

@shell(single=True)
def multisudo(spec, arg):
    """run the sudoed command in parallel on the various hosts"""

    def run_sudo():
        return sudo(
            arg, spec.shell, spec.pty, spec.combine_stderr, None, spec.dir,
            spec.format
            )

    env().multirun(
        run_sudo, spec.shell, spec.pty, spec.combine_stderr, spec.dir,
        spec.format, quiet_exit=1
        )

@shell(single=True)
def help(spec, arg):
    """display the list of available builtin commands"""

    max_len = max(len(x) for x in SHELL_BUILTINS)
    max_width = 80 - max_len - 5
    print
    print "Available Builtins:"
    print
    for builtin in sorted(SHELL_BUILTINS):
        padding = (max_len - len(builtin)) * ' '
        docstring = SHELL_BUILTINS[builtin].__doc__ or ''
        if len(docstring) > max_width:
            docstring = docstring[:max_width-3] + "..."
        print "  %s%s   %s" % (padding, builtin, docstring)
    print

# ------------------------------------------------------------------------------
# Monkey-Patch The Global Env Object
# ------------------------------------------------------------------------------

env.__dict__['_env_mgr'] = EnvManager
env.__dict__['_ctx_class'] = ContextRunner
env.get_settings = get_settings

def __env_getattr__(self, key):
    if key.isupper():
        return self._env_mgr.for_var(key)
    try:
        return self[key]
    except KeyError:
        raise AttributeError(key)

def __env_call__(self, *args, **kwargs):
    return self._ctx_class(*args, **kwargs)

env.__class__.__getattr__ = __env_getattr__
env.__class__.__call__ = __env_call__

# ------------------------------------------------------------------------------
# Readline Completer
# ------------------------------------------------------------------------------

binaries_on_path = []

def get_binaries_on_path():
    env_path = environ.get('PATH')
    if not env_path:
        return
    append = binaries_on_path.append
    for path in env_path.split(pathsep):
        path = path.strip()
        if not path:
            continue
        if not isdir(path):
            continue
        for file in listdir(path):
            file_path = join(path, file)
            if access(file_path, X_OK):
                append(file)
    binaries_on_path.sort()

def complete(text, state, matches=[], binaries={}):
    if not state:
        if text.startswith('.'):
            text = text[1:]
            matches[:] = [
                '.' + builtin + ' '
                for builtin in SHELL_BUILTINS if builtin.startswith(text)
                ]
        elif text.startswith('{'):
            text = text[1:]
            matches[:] = [
                '{' + prop + '}'
                for prop in env if prop.startswith(text)
                ]
        else:
            if not binaries_on_path:
                get_binaries_on_path()
            matches[:] = []; append = matches.append
            for file in binaries_on_path:
                if file.startswith(text):
                    append(file)
                else:
                    if matches:
                        break
    try:
        return matches[state]
    except IndexError:
        return

try:
    import readline
except ImportError:
    readline = None
else:
    readline.set_completer_delims(' \t\n')
    readline.set_completer(complete)
    readline.parse_and_bind('tab: complete')

# ------------------------------------------------------------------------------
# Task Decorator
# ------------------------------------------------------------------------------

def task(*args, **kwargs):
    """Decorate a callable as being a task."""

    display = kwargs.get('display', 1)
    if args:
        if hasattr(args[0], '__call__'):
            func = args[0]
            func.__task__ = 1
            if not display:
                func.__hide__ = 1
            return func
        ctx = args
        if len(ctx) == 1 and not isinstance(ctx[0], basestring):
            ctx = tuple(args[0])
    else:
        ctx = ()

    def __task(__func):
        __func.__ctx__ = ctx
        __func.__task__ = 1
        if not display:
            __func.__hide__ = 1
        return __func

    return __task

# ------------------------------------------------------------------------------
# Stages Support
# ------------------------------------------------------------------------------

def set_env_stage_command(tasks, stage):
    if stage in tasks:
        return
    def set_stage():
        """Set the environment to %s.""" % stage
        puts('env.stage = %s' % stage, 'system')
        env.stage = stage
        config_file = env.config_file
        if config_file:
            if not isinstance(config_file, basestring):
                config_file = '%s.yaml'
            try:
                env.config_file = config_file % stage
            except TypeError:
                env.config_file = config_file
    set_stage.__hide__ = 1
    set_stage.__name__ = stage
    set_stage.__task__ = 1
    tasks[stage] = set_stage
    return set_stage

# ------------------------------------------------------------------------------
# Global Defaults Initialisation
# ------------------------------------------------------------------------------

def setup_defaults(path=None):
    """Initialise ``env`` and ``output`` to default values."""

    env.update({
        'again_prompt': 'Sorry, try again.',
        'always_use_pty': True,
        'colors': False,
        'color_settings': {
            'abort': yellow,
            'error': yellow,
            'finish': cyan,
            'host_prefix': green,
            'prefix': red,
            'prompt': blue,
            'task': red,
            'warn': yellow
            },
        'combine_stderr': True,
        'command': None,
        'command_prefixes': [],
        'config_file': None,
        'ctx': (),
        'cwd': '',
        'disable_known_hosts': False,
        'echo_stdin': True,
        'hook': None,
        'host': None,
        'host_string': None,
        'key_filename': None,
        'lcwd': '',
        'multirun_child_timeout': 10,
        'multirun_pool_size': 10,
        'no_agent': False,
        'no_keys': False,
        'output_prefix': True,
        'password': None,
        'passwords': {},
        'port': None,
        'reject_unknown_hosts': False,
        'shell': '/bin/bash -l -c',
        'shell_history_file': '~/.bolt-shell-history',
        'sudo_prefix': "sudo -S -p '%s' ",
        'sudo_prompt': 'sudo password:',
        'use_shell': True,
        'user': _get_system_username(),
        'warn_only': False
        })

    output.update({
        'aborts': True,
        'debug': False,
        'running': True,
        'status': True,
        'stderr': True,
        'stdout': True,
        'user': True,
        'warnings': True,
        }, {
        'everything': ['output', 'running', 'user', 'warnings'],
        'output': [ 'stderr', 'stdout']
        })

    # Load defaults from a YAML file.
    if path and exists(path):
        fileobj = open(path, 'rb')
        mapping = load_yaml(fileobj.read())
        if not isinstance(mapping, dict):
            abort(
                "Got a %r value when loading %r. Mapping expected." %
                (type(mapping), path)
                )
        env.update(mapping)

# ------------------------------------------------------------------------------
# Task Runner Initialiser
# ------------------------------------------------------------------------------

def init_task_runner(filename, cwd):
    """Return a TaskRunner initialised from the located Boltfile."""

    cwd = abspath(cwd)
    if sep in filename:
        path = join(cwd, filename)
        if not exists(path):
            abort("Couldn't find Boltfile: %s" % filename)
    else:
        prev = None
        while cwd:
            if cwd == prev:
                abort("Couldn't find Boltfile: %s" % filename)
            path = join(cwd, filename)
            if exists(path):
                break
            prev = cwd
            cwd = dirname(cwd)

    directory = dirname(path)
    if directory not in sys.path:
        sys.path.insert(0, directory)

    chdir(directory)

    sys.dont_write_bytecode = 1
    boltfile = load_source('boltfile', path)
    sys.dont_write_bytecode = 0

    tasks = dict(
        (var.replace('_', '-'), obj) for var, obj in vars(boltfile).items()
        if hasattr(obj, '__task__')
        )

    stages = environ.get('BOLT_STAGES', env.get('stages'))
    if stages:
        if isinstance(stages, basestring):
            stages = [stage.strip() for stage in stages.split(',')]
        env.stages = stages
        for stage in stages:
            set_env_stage_command(tasks, stage)

    return TaskRunner(directory, path, boltfile.__doc__, tasks)

# ------------------------------------------------------------------------------
# Task Runner
# ------------------------------------------------------------------------------

class TaskRunner(object):
    """Task runner encapsulation."""

    def __init__(self, directory, path, docstring, tasks):
        self.directory = directory
        self.path = path
        self.docstring = docstring
        self.tasks = tasks

    def display_listing(self):
        """Print a listing of the available tasks."""

        docstring = self.docstring
        if docstring:
            docstring = docstring.strip()
            if docstring:
                print(docstring + '\n')

        print("Available tasks:\n")

        tasks = self.tasks
        width = max(map(len, tasks)) + 3

        for name in sorted(tasks):
            task = tasks[name]
            if hasattr(task, '__hide__'):
                continue
            padding = " " * (width - len(name))
            print("    %s%s%s" % (name, padding, (task.__doc__ or "")[:80]))
        print

        if 'stages' in env:
            print 'Available environments:'
            print
            for stage in env.stages:
                print '    %s' % stage
            print

        call_hooks('listing.display')

    def execute_task(self, name, args, kwargs, ctx):
        """Execute the given task."""

        task = self.tasks[name]
        env.command = name
        if output.running:
            msg = "running task: %s" % name
            prefix = '[system] '
            if env.colors:
                prefix = env.color_settings['prefix'](prefix)
            print(prefix + msg)

        if not ctx:
            ctx = getattr(task, '__ctx__', None)

        if ctx:
            with settings(ctx=ctx):
                task(*args, **kwargs)
            return

        task(*args, **kwargs)

    def run(self, spec):
        """Execute the various tasks given in the spec list."""

        try:

            if output.debug:
                names = ", ".join(info[0] for info in spec)
                print("Tasks to run: %s" % names)

            call_hooks('commands.before', self.tasks, spec)

            # Initialise the default stage if none are given as the first task.
            if 'stages' in env:
                if spec[0][0] not in env.stages:
                    self.execute_task(env.stages[0], (), {}, None)
                else:
                    self.execute_task(*spec.pop(0))

            # Load the config YAML file if specified.
            if env.config_file:
                config_path = realpath(expanduser(env.config_file))
                config_path = join(self.directory, config_path)
                config_file = open(config_path, 'rb')
                config = load_yaml(config_file.read())
                if not config:
                    env.config = AttrDict()
                elif not isinstance(config, dict):
                    abort("Invalid config file found at %s" % config_path)
                else:
                    env.config = AttrDict(config)
                config_file.close()

            call_hooks('config.loaded')

            # Execute the tasks in order.
            for info in spec:
                self.execute_task(*info)

            if output.status:
                msg = "\nDone."
                if env.colors:
                    msg = env.color_settings['finish'](msg)
                print(msg)

        except SystemExit:
            raise
        except KeyboardInterrupt:
            if output.status:
                msg = "\nStopped."
                if env.colors:
                    msg = env.color_settings['finish'](msg)
                print >> sys.stderr, msg
            sys.exit(1)
        except:
            sys.excepthook(*sys.exc_info())
            sys.exit(1)
        finally:
            call_hooks('commands.after')
            disconnect_all()

# ------------------------------------------------------------------------------
# Script Runner
# ------------------------------------------------------------------------------

def main(argv=None):
    """Handle the bolt command line call."""

    if argv is None:
        argv = sys.argv[1:]

    op = OptionParser(
        usage="bolt <command-1> <command-2> ... [options]",
        )

    op.add_option(
        '-v', '--version', action='store_true', default=False,
        help="show program's version number and exit"
        )

    op.add_option(
        '-f', dest='file', default="Boltfile",
        help="set the name or path of the bolt file [Boltfile]"
        )

    op.add_option(
        '-d', dest='defaults_file', default=_rc_path(),
        help="set the path of the defaults file [~/.bolt.yaml]"
        )

    op.add_option(
        '-i',  dest='identity', action='append', default=None,
        help="path to SSH private key file(s) -- may be repeated"
        )

    op.add_option(
        '--hide', metavar='LEVELS',
        help="comma-separated list of output levels to hide"
        )

    op.add_option(
        '--show', metavar='LEVELS',
        help="comma-separated list of output levels to show"
        )

    op.add_option(
        '--disable', metavar='HOOKS',
        help="comma-separated list of hooks to disable"
        )

    op.add_option(
        '--enable', metavar='HOOKS',
        help="comma-separated list of hooks to enable"
        )

    op.add_option(
        '--list', action='store_true', default=False,
        help="show the list of available tasks and exit"
        )

    op.add_option(
        '--no-pty', action='store_true', default=False,
        help="do not use pseudo-terminal in run/sudo"
        )

    options, args = op.parse_args(argv)
    setup_defaults(options.defaults_file)

    # Load the Boltfile.
    runner = init_task_runner(options.file, getcwd())

    # Autocompletion support.
    autocomplete_items = runner.tasks.keys()
    if 'autocomplete' in env:
        autocomplete_items += env.autocomplete

    autocomplete(op, ListCompleter(autocomplete_items))

    if options.version:
        print("bolt %s" % __version__)
        sys.exit()

    if options.no_pty:
        env.always_use_pty = False

    if options.identity:
        env.key_filename = options.identity

    split_string = lambda s: filter(None, map(str.strip, s.split(',')))

    # Handle output levels.
    if options.show:
        for level in split_string(options.show):
            output[level] = True

    if options.hide:
        for level in split_string(options.hide):
            output[level] = False

    if output.debug:
        print("Using Boltfile: %s" % runner.path)

    # Handle hooks related options.
    if options.disable:
        for hook in split_string(options.disable):
            DISABLED_HOOKS.append(hook)

    if options.enable:
        for hook in split_string(options.enable):
            ENABLED_HOOKS.append(hook)

    if options.list:
        print('\n'.join(sorted(runner.tasks)))
        sys.exit()

    tasks = []
    idx = 0

    # Parse command line arguments.
    for task in args:

        # Initialise variables.
        _args = []
        _kwargs = {}
        _ctx = None

        # Handle +env flags.
        if task.startswith('+'):
            if ':' in task:
                name, value = task[1:].split(':', 1)
                env[name] = value
            else:
                env[task[1:]] = True
            continue

        # Handle @context specifiers.
        if task.startswith('@'):
            if not idx:
                continue
            ctx = (task[1:],)
            existing = tasks[idx-1][3]
            if existing:
                new = list(existing)
                new.extend(ctx)
                ctx = tuple(new)
            tasks[idx-1][3] = ctx
            continue

        # Handle tasks with parameters.
        if ':' in task:
            task, argstr = task.split(':', 1)
            for pair in _escape_split(',', argstr):
                k, _, v = pair.partition('=')
                if _:
                    _kwargs[k] = v
                else:
                    _args.append(k)

        idx += 1
        task_name = task.replace('_', '-')

        if task_name not in runner.tasks:
            abort("Task not found:\n\n%s" % indent(task))

        tasks.append([task_name, _args, _kwargs, _ctx])

    if not tasks:
        runner.display_listing()
        sys.exit()

    runner.run(tasks)

# ------------------------------------------------------------------------------
# Self Runner
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
