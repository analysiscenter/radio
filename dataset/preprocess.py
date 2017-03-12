""" Preprocessing """


def action(method):
    """ Decorator for action methods in Batch classes """
    # TODO: decorator params: parallelization (e.g. threads, processes, async/await, greenlets,...)
    # use __action for class-specific params
    method._action = True # pylint: disable=protected-access
    return method


class Preprocessing:
    """ Preprocessing """
    def __init__(self, dataset):
        self.dataset = dataset
        self.action_list = []
        self.batch_generator = None


    def __getattr__(self, name, *args, **kwargs):
        """ Check if an unknown attr is an action from the batch class """
        if hasattr(self.dataset.batch_class, name):
            attr_name = getattr(self.dataset.batch_class, name)
            if callable(attr_name):
                try:
                    if hasattr(attr_name, "_action"):
                        self.action_list.append({'name': name})
                except AttributeError:
                    raise ValueError("Method %s is not marked with @action decorator" % name)
        else:
            raise AttributeError("Method %s has not been found in Preprocessing and Batch classes" % name)
        return self.append_action


    def append_action(self, *args, **kwargs):
        """ Add new action to the log of future actions """
        last = len(self.action_list) - 1
        self.action_list[last].update({'args': args, 'kwargs': kwargs})
        return self


    def _exec_all_actions(self, batch):
        for _action in self.action_list:
            batch_action = getattr(batch, _action['name'])
            batch_action(*_action['args'], **_action['kwargs'])


    def _run_seq(self, gen_batch):
        for batch in gen_batch:
            self._exec_all_actions(batch)


    def run(self, batch_size, shuffle=False, *args, **kwargs):
        """ Execute all lazy actions for each batch in the dataset
            Batches are created sequentially, one after another, without batch-level parallelism
        """
        batch_generator = self.dataset.gen_batch(batch_size, shuffle=shuffle, one_pass=True, *args, **kwargs)
        self._run_seq(batch_generator)
        return self


    def create_batch(self, batch_indices, *args, **kwargs):
        """ Create a new batch by give indices and execute all previous lazy actions """
        batch = self.dataset.create_batch(batch_indices, *args, **kwargs)
        self._exec_all_actions(batch)
        return batch


    def next_batch(self, batch_size, shuffle=False, one_pass=False, *args, **kwargs):
        """ Get the next batch and execute all previous lazy actions """
        if self.batch_generator is None:
            self.batch_generator = self.dataset.gen_batch(batch_size, shuffle=shuffle,
                                                          one_pass=one_pass, *args, **kwargs)
        batch = next(self.batch_generator)
        self._exec_all_actions(batch)
        return batch
