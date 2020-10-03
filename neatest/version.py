class Version():
    def __init__(self):
        self.major: int = 1
        self.minor: int = 0
        self.patch: int = 1

    def __eq__(self, other: 'Version') -> bool:
        return (self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch)

VERSION: Version = Version()
