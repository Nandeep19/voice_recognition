from fastapi import UploadFile


async def read_upload(upload: UploadFile) -> tuple[str, bytes]:
    """(filename, bytes) of an uploaded file, the form the service takes."""
    return upload.filename or "clip", await upload.read()
