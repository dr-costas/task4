import numpy
import visdom
from mimir.handlers import Handler


class VisdomHandler(Handler):
    r"""A mimir logger handler for plotting with visdom.

    The log should contain two fields: `'iteration'` (integer) and
    `'records'` (dict). The records dictionary has a form
    `{line_name: {key: value}}`, where the line name is in the `line_names`.
    The handler ignores extra content in the log. Multiple handlers
    are expected to be used to create multiple plot windows.

    Parameters
    ----------
    line_names : list of str
        Line names to be shown in legend. Same names should appear
        in the log.
    key : str
        Key to be plotted.
    plot_options : dict
        Plot options are passed as `opts` argument to the
        :meth:`Visdom.line`.
    filters : iterable
        Filters to be passed to the :class:`Handler`.
    \*\*kwargs : dict
        Keyword arguments to be passed to the :class:`Visdom` constructor.

    Examples
    --------
    >>> import mimir
    >>> logger = mimir.Logger()
    >>> visdom_handler = VisdomHandler(
    ...     ['train', 'valid'], 'ce',
    ...     dict(title='Train/valid cross-entropy',
    ...          xlabel='iteration',
    ...          ylabel='cross-entropy'))
    >>> logger.handlers.append(visdom_handler)
    >>> logger.log({'iteration': 1,
    ...             'records': {'train': {'ce': 3.14},
    ...                         'valid': {'ce': 6.28}}})

    """
    def __init__(self, line_names, key, plot_options=None, filters=None,
                 **kwargs):
        super(VisdomHandler, self).__init__(filters=filters)
        self.viz = visdom.Visdom(**kwargs)
        self.line_names = line_names
        self.key = key

        # we have to initialize the plot with some data, but NaNs are ignored
        dummy_data = [numpy.nan] * len(self.line_names)
        dummy_ind = [0.] * len(self.line_names)
        plot_options.update(dict(legend=line_names))
        # `line` method squeezes the input, in order to maintain the shape
        # we have to repeat it twice making its shape (2, M), where M is
        # the number of lines
        self.window = self.viz.line(numpy.vstack([dummy_data, dummy_data]),
                                    numpy.vstack([dummy_ind, dummy_ind]),
                                    opts=plot_options)

    def log(self, entity):
        iteration = entity['iteration']
        records = entity['records']
        for name in self.line_names:
            if name in records:
                values = numpy.array(
                    [records[name][self.key]], dtype='float64')
                iterations = numpy.array([iteration], dtype='float64')
                self.viz.updateTrace(iterations, values, append=True,
                                     name=name, win=self.window)