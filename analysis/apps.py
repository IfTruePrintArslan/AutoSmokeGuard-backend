"""
App configuration for the analysis (detection + segmentation) domain.

``ready()`` is the only startup hook Django offers, so it is where the
in-process job worker gets its two boot-time chores:

1. **Recovery.**  An in-process queue does not survive a restart.  Rows left
   ``running`` or ``queued`` by a crashed process have nothing behind them and
   would otherwise show a progress bar that never moves.

   The sweep resolves only the rows it can *prove* are abandoned: it is
   scoped by the ``(host, pid, boot)`` stamp every non-terminal row carries.
   That scoping is not optional — the shipped production default is
   ``gunicorn --workers 2`` (``docker/entrypoint.sh``), so ``ready()`` runs
   once per worker process, and a worker the arbiter has just replaced must
   never resolve its sibling's live analysis.  See
   :func:`analysis.worker.recover_interrupted_jobs`.

2. **Warm-up.**  Loading YOLO and the U-Net takes several seconds.  Doing it
   on a daemon thread at boot means the first user to press Analyse waits for
   inference, not for imports.

Both run on a background thread and both are heavily guarded — ``ready()``
also fires for ``manage.py migrate`` (against a database that may have no
tables yet), for ``manage.py check`` and under pytest, and in none of those
cases should the process load torch or touch job rows.  See
:func:`analysis.worker.should_bootstrap` for which invocations count as
serving (including ``runserver --noreload``, gunicorn and uWSGI), and
:func:`analysis.worker.start_bootstrap` for the per-process idempotency
guard that stops a second ``ready()`` from starting a second boot thread.
"""
import logging

from django.apps import AppConfig

logger = logging.getLogger('asg.worker')


class AnalysisConfig(AppConfig):
    """App config for detection / segmentation results."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analysis'
    verbose_name = 'Analysis'

    def ready(self):
        """Start the worker's boot thread when this process serves traffic."""
        # Imported here, not at module scope: ``apps.py`` is imported during
        # app-registry population, before models are loadable.
        from . import worker

        try:
            if worker.start_bootstrap() is not None:
                logger.info('Analysis worker bootstrap thread started')
        except Exception:                             # noqa: BLE001
            # A failed bootstrap must never stop the site from starting; the
            # API is useful without the ML runtime.
            logger.exception('Analysis worker bootstrap could not be started')
