import os
from nz_data_challenge import submit_utils


def setup_public_area() -> None:
    """
    A function download the public data
    """

    if not os.path.exists("public"):
        # Note that the tar file has "public" as top level directory
        # so we if we extract to "tests" the files actually end
        # up in "tests/public"
        submit_utils.download_and_extract_tar(submit_utils.PUBLIC_URL, ".")


if __name__ == '__main__':

    setup_public_area()
