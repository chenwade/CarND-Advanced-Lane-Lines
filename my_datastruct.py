import numpy as np
# 自定义循环队列类
class MyCirculateQueue(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize  # 定义队列长度
        self.queue = []  # 存储队列 列表

    def __str__(self):
        # 返回对象的字符串表达式，方便查看
        return str(self.queue)

    def is_empty(self):
        return self.queue == []

    def is_full(self):
        return self.size() == self.maxsize

    def enqueue(self, item):
        if self.is_full():
            self.dequeue()
        self.queue.insert(0, item)

    def dequeue(self):
        if self.is_empty():
            return None
        return self.queue.pop()

    def find(self, value):
        # if find value in the queue, return the index, or return None
        for i in range(len(self.queue)):
            if self.queue[-1 - i] == value:
                return i
        return None

    def visit(self, index):
        # return the value of the index of queue
        assert 0 <= index < len(self.queue)
        return self.queue[-1 - index]


    def get_tail(self):
        if self.is_empty():
            return None
        return self.queue[0]

    def size(self):
        return len(self.queue)



if __name__ == "__main__":
    a = np.array([1, 2, 300])
    a[2] += 750

    my_queue = MyCirculateQueue(5)
    for i in range(10):
        my_queue.enqueue(i)
    for i in range(10):
        my_queue.dequeue()
    for i in range(5):
        my_queue.enqueue(i)
    value = my_queue.visit(5)
    index = my_queue.find(4)
    c = 1
