import time

class DataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.last_waiting_time = 0

    def __iter__(self):
        dataloader_iter = iter(self.dataloader)

        while True:
            start = time.perf_counter()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            self.last_waiting_time = time.perf_counter() - start
            yield batch

    def __len__(self):
        return len(self.dataloader)
