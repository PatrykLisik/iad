
class Fixed_size_queue:
    """
    Data structure that contains const number of element even after append
    """

    def __init__(self, size):

        # legth of queue
        self.size = size
        self._data = [None] * size

    def append(self, obj):
        """
        Append element to queue and delets the lase one
        """
        self._data.insert(-1, obj)
        self._data.pop()

    def contains(self, obj):
        """
        Checks if queue contains element equal to given
        """
        return obj in self._data
