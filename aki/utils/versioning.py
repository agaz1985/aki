import logging


class Version:
    def __init__(self, version: str):
        self._major = 0
        self._minor = 0
        self._patch = 0
        self.set(version)

    def set(self, version_string: str):
        self._major, self._minor, self._patch = version_string.split('.')

    def get_as_string(self) -> str:
        return '.'.join(self.get_as_array())

    def get_as_array(self):
        return [self._major, self._minor, self._patch]

    def compare(self, version):
        """
        Compare the current version with the input one.
        :param version: version to compare the current with.
        :return:
        1 if the current version is higher than the input,
        -1 if the current version is lower than the input.
        0 if the two versions are the same.
        """
        major, minor, patch = version.get_as_array()
        if self._major > major:
            return 1
        elif self._major < major:
            return -1
        else:
            if self._minor > minor:
                return 1
            elif self._minor < minor:
                return -1
            else:
                if self._patch > patch:
                    return 1
                elif self._patch < patch:
                    return -1
                else:
                    return 0


# Define the current aki version number.
AKI_VERSION = "0.1.0"

CURRENT_VERSION = Version(AKI_VERSION)

logger = logging.getLogger('aki_logger')
logger.info(f"aki v.{CURRENT_VERSION.get_as_string()}.")
