from metaflow import FlowSpec, step

class SampleFlow(FlowSpec):

    @step
    def start(self):
        self.my_data = 'Hello, Metaflow!'
        self.next(self.end)

    @step
    def end(self):
        print(f'The data is: {self.my_data}')

if __name__ == '__main__':
    SampleFlow()
