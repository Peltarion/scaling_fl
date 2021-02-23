from typing import Optional

from sacred.observers import MongoObserver


class ResumableMongoObserver(MongoObserver):
    """MongoObserver that allows resuming runs based on a key in the config

    Args:
        resume_key: Key in config containing the id to use for matching experiments
        *args, **kwargs: Forwarded to MongoObserver
    """
    # Keys to persist over runs
    PERSIST_KEYS = [
        'artifacts',
        'captured_out',
        'resources',
        'start_time'
    ]

    def __init__(self, resume_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._persisted = {}
        self._previous_captured_out = None
        self._resume_key = resume_key
        self._resume_id = None

    def _load_previous(self, resume_id: str):
        """Load previous run entry."""
        self._resume_id = resume_id

        # Find and load previous entry if any
        previous_run_entry = self._find_previous(resume_id)
        if previous_run_entry is not None:
            self.overwrite = previous_run_entry
            self._previous_captured_out = previous_run_entry['captured_out']
            # Persist data after resume
            for key in self.PERSIST_KEYS:
                self._persisted[key] = previous_run_entry[key]

    def _find_previous(self, resume_id: str) -> Optional[dict]:
        """Return the last run_entry associated with the resume key.

        Raises:
            ValueError: If resume_id in multiple runs
        """
        cursor = self.runs.find({f'config.{self._resume_key}': resume_id})
        run_entry = None
        for i, entry in enumerate(cursor):
            if i > 0:
                raise ValueError('Cannot resume job, there exists multiple runs '
                                 f'with the config.{self._resume_key}={resume_id}')
            run_entry = entry
        return run_entry

    def started_event(self,
                      ex_info,
                      command,
                      host_info,
                      start_time,
                      config,
                      meta_info,
                      _id):
        """Start event, add resume_key and persist previous data."""
        # Store previous
        resume_id = None
        if self._resume_key is not None:
            resume_id = config.get(self._resume_key)
            if resume_id is not None:
                self._load_previous(resume_id)

        result = super().started_event(
            ex_info, command, host_info, start_time, config, meta_info, _id)

        # Restore entries and save in db
        if resume_id is not None:
            for key, value in self._persisted.items():
                self.run_entry[key] = value
            if self._persisted:
                self.save()
        return result

    def heartbeat_event(self, info, captured_out, beat_time, result):
        """Hearbeat event that persists previous capture."""
        super().heartbeat_event(
            info,
            self._persisted.get('captured_out', '') + captured_out,
            beat_time,
            result
        )
