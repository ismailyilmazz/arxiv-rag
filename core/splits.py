import hashlib

TEST_PERCENT = 1


def split_of(paper_id: str) -> str:
    bucket = int(hashlib.md5(paper_id.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if bucket < TEST_PERCENT else "train"
