
class CleanExit:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            print('user interrupt')
            return True
        return exc_type is None

    
class Pipeline:
    
    def __init__(self):
        pass
        
    def run(self):
        raise NotImplementedError
        



