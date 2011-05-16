Bolt lets you easily automate sysadmin tasks like deployment. You can use it to
manage multi-server setups over SSH or even as a build tool. To use, simply
create a ``Boltfile`` with your tasks, e.g.

::

    from bolt.api import *

    @task
    def deploy():
        """publish the latest version of the app"""

        with cd('/var/www/mysite.com'):
            run('git remote update')
            run('git checkout origin/master')

        sudo("/etc/init.d/apache2 graceful")

And then, run the tasks from the command line, e.g.

::

    $ bolt deploy

Bolt was initially developed as a fork of `Fabric <http://fabfile.org/>`_, but
has since been extracted as a standalone tool without any of the historic
baggage of the Fabric APIs.

**Documentation**

Bolt doesn't currently have any docs, but you can look at the introduction to
the Fabric fork for details of how to use most of its features. Simply replace
the references to ``fab`` and ``fabric`` with ``bolt``:

* `Fabric with Cleaner API and Parallel Deployment Support
  <http://tav.espians.com/fabric-python-with-cleaner-api-and-parallel-deployment-support.html>`_

**Contribute**

To contribute any patches simply fork the repository on GitHub and send a pull
request to https://github.com/tav, thanks!

**License**

The code derived from Fabric is contained within the ``bolt/fabcode.py`` file
and is under the BSD license. The rest of the code has been released into the
`Public Domain <https://github.com/tav/bolt/raw/master/UNLICENSE>`_. Do with it
as you please.

-- 
Enjoy, tav <tav@espians.com>
