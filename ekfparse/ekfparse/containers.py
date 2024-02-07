
class DefDict (dict):
    def set_default ( self, value ):
        self.default = value
    def __getitem__(self, __key):
        if __key in self.keys():
            return super().__getitem__(__key)
        else:
            return self.default