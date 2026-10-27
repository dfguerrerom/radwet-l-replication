"""Google Earth Engine export and download utilities.

  - init_gee(): initialise Earth Engine with a Cloud project
  - export_and_download_batch(): submit exports, poll, download from Drive

Uses the standard ee library for task submission and monitoring, and
google-api-python-client for Drive downloads. Drive credentials are
derived from the EE credentials file (~/.config/earthengine/credentials).

If the EE credentials lack Drive scope, the function will still submit and
monitor all export tasks, then print instructions for manual download.

To enable automatic Drive downloads, re-authenticate with Drive scope:
  earthengine authenticate --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.full_control
"""

import io
import json
import os
import time
from pathlib import Path

import ee

DEFAULT_MAX_PIXELS = 1e13
DEFAULT_FILE_FORMAT = "GeoTIFF"
POLL_INTERVAL = 30  # seconds between status checks
PROJECT_ENV_VAR = "EE_PROJECT"


def init_gee(project=None):
    """Initialise Google Earth Engine.

    The Cloud project is taken from ``project``, then the ``EE_PROJECT``
    environment variable, then the default stored by
    ``earthengine set_project``.

    Raises:
        ee.EEException: If authentication fails. Run
            'earthengine authenticate' first.
    """
    if ee.data.is_initialized():
        return
    project = project or os.environ.get(PROJECT_ENV_VAR) or None
    ee.Initialize(project=project)


def _build_drive_service():
    """Build a Google Drive API service using EE credentials.

    Reads the refresh token from the EE credentials file and uses the
    EE OAuth client ID/secret to obtain an access token. This works only
    if the credentials were created with Drive scope.

    Returns:
        googleapiclient.discovery.Resource or None: Drive service, or None
            if credentials lack Drive scope or dependencies are missing.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        print("WARNING: google-api-python-client or google-auth not installed.")
        print("  Drive download disabled. Install with:")
        print("  pip install google-api-python-client google-auth")
        return None

    cred_path = Path.home() / ".config" / "earthengine" / "credentials"
    if not cred_path.exists():
        print(f"WARNING: EE credentials not found at {cred_path}")
        return None

    data = json.loads(cred_path.read_text())
    refresh_token = data.get("refresh_token")
    if not refresh_token:
        print("WARNING: No refresh_token in EE credentials file.")
        return None

    # Use EE's OAuth client ID and secret
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=ee.oauth.CLIENT_ID,
        client_secret=ee.oauth.CLIENT_SECRET,
    )

    try:
        creds.refresh(Request())
    except Exception as exc:
        print(f"WARNING: Could not refresh credentials for Drive access: {exc}")
        return None

    # Test that Drive scope is available
    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        service.files().list(pageSize=1, fields="files(id)").execute()
        return service
    except Exception as exc:
        if "insufficientPermissions" in str(exc) or "403" in str(exc):
            print("WARNING: EE credentials lack Google Drive scope.")
            print("  To enable automatic downloads, re-authenticate with:")
            print(
                "  earthengine authenticate"
                " --scopes=https://www.googleapis.com/auth/earthengine"
                ",https://www.googleapis.com/auth/drive"
                ",https://www.googleapis.com/auth/cloud-platform"
                ",https://www.googleapis.com/auth/devstorage.full_control"
            )
        else:
            print(f"WARNING: Drive API error: {exc}")
        return None


def _find_drive_file(service, filename, folder_name=None):
    """Find a file on Google Drive by name, optionally within a folder.

    Args:
        service: Google Drive API service.
        filename: Name of the file (without .tif extension -- GEE may add it).
        folder_name: Optional Drive folder name to search within.

    Returns:
        str or None: File ID if found, None otherwise.
    """
    candidates = [filename, f"{filename}.tif"]

    for name in candidates:
        query = f"name = '{name}' and trashed = false"

        if folder_name:
            folder_query = (
                f"name = '{folder_name}'"
                " and mimeType = 'application/vnd.google-apps.folder'"
                " and trashed = false"
            )
            folder_results = (
                service.files()
                .list(q=folder_query, fields="files(id)", pageSize=5)
                .execute()
            )
            folders = folder_results.get("files", [])
            if folders:
                folder_id = folders[0]["id"]
                query += f" and '{folder_id}' in parents"

        results = (
            service.files()
            .list(q=query, fields="files(id, name)", pageSize=5)
            .execute()
        )
        files = results.get("files", [])
        if files:
            return files[0]["id"]

    return None


def _download_drive_file(service, file_id, output_path):
    """Download a file from Google Drive to a local path.

    Args:
        service: Google Drive API service.
        file_id: Google Drive file ID.
        output_path: Local file path to save to.

    Returns:
        bool: True if download succeeded.
    """
    from googleapiclient.http import MediaIoBaseDownload

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"    Download progress: {pct}%", end="\r")

    with open(output_path, "wb") as f:
        f.write(fh.getvalue())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    Downloaded: {output_path} ({size_mb:.1f} MB)")
    return True


def _delete_drive_file(service, file_id, filename):
    """Delete a file from Google Drive.

    Args:
        service: Google Drive API service.
        file_id: Google Drive file ID.
        filename: Filename (for logging).
    """
    try:
        service.files().delete(fileId=file_id).execute()
        print(f"    Deleted from Drive: {filename}")
    except Exception as exc:
        print(f"    WARNING: Could not delete {filename} from Drive: {exc}")


def _format_elapsed(seconds):
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}hr"


def export_and_download_batch(exports, folder="radwet_l"):
    """Submit GEE export tasks, monitor progress, and download results.

    This function is blocking -- it submits all tasks, waits for all to
    complete (or fail), then downloads results from Google Drive.

    Args:
        exports: List of export dicts, each containing:
            - image (ee.Image): Image to export
            - description (str): GEE task description
            - filename (str): Drive filename prefix (no extension)
            - output_path (str): Local path to save the downloaded file
            - scale (int/float): Export scale in meters
            - region (ee.Geometry): Export region
        folder: Google Drive folder name for exports.

    The function prints progress updates throughout. If Drive download
    is not available (missing scope), it still submits and monitors tasks,
    then prints instructions for manual download.
    """
    if not exports:
        print("No exports to process.")
        return

    n = len(exports)
    print(f"\nSubmitting {n} export task(s) to Google Drive folder '{folder}/'...")
    print()

    # --- Phase 1: Submit all export tasks ---
    tasks = []
    for i, export in enumerate(exports):
        desc = export["description"]
        print(f"  [{i+1}/{n}] Submitting: {desc}")

        task = ee.batch.Export.image.toDrive(
            image=export["image"],
            description=desc,
            fileNamePrefix=export["filename"],
            folder=folder,
            scale=export["scale"],
            region=export["region"],
            maxPixels=DEFAULT_MAX_PIXELS,
            fileFormat=DEFAULT_FILE_FORMAT,
            # Optional explicit CRS. Defaults to None (image's native projection,
            # preserving prior behavior). Pass crs="EPSG:4326" to force lon/lat
            # export with nearest-neighbor resampling (matches the validated
            # `near` reprojection for SAR DN data).
            crs=export.get("crs"),
        )
        task.start()

        tasks.append({
            "task": task,
            "task_id": task.id,
            "export": export,
        })
        print(f"           Task ID: {task.id}")

    print()
    print(f"All {n} tasks submitted. Monitoring progress...")
    print()

    # --- Phase 2: Poll until all tasks complete ---
    start_time = time.time()
    completed = set()
    failed = {}

    while len(completed) + len(failed) < n:
        time.sleep(POLL_INTERVAL)

        for i, t in enumerate(tasks):
            if i in completed or i in failed:
                continue

            task_id = t["task_id"]
            desc = t["export"]["description"]

            try:
                statuses = ee.data.getTaskStatus(task_id)
                if not statuses:
                    continue
                status = statuses[0]
                state = status.get("state", "UNKNOWN")
            except Exception as exc:
                print(f"  WARNING: Could not get status for {desc}: {exc}")
                continue

            if state == "COMPLETED":
                completed.add(i)
                elapsed = _format_elapsed(time.time() - start_time)
                print(
                    f"  COMPLETED [{len(completed)+len(failed)}/{n}]:"
                    f" {desc} ({elapsed})"
                )
            elif state == "FAILED":
                error_msg = status.get("error_message", "Unknown error")
                failed[i] = error_msg
                print(f"  FAILED [{len(completed)+len(failed)}/{n}]: {desc}")
                print(f"    Error: {error_msg}")
            elif state in ("CANCEL_REQUESTED", "CANCELLED"):
                failed[i] = "Cancelled"
                print(f"  CANCELLED [{len(completed)+len(failed)}/{n}]: {desc}")

        pending = n - len(completed) - len(failed)
        if pending > 0:
            elapsed = _format_elapsed(time.time() - start_time)
            print(f"  ... [{elapsed}] {pending} task(s) still running", flush=True)

    total_elapsed = _format_elapsed(time.time() - start_time)
    print()
    print(f"Task summary: {len(completed)} completed, {len(failed)} failed"
          f" ({total_elapsed})")

    if not completed:
        print("No tasks completed successfully. Nothing to download.")
        if failed:
            print("\nFailed tasks:")
            for i, error_msg in failed.items():
                print(f"  {tasks[i]['export']['description']}: {error_msg}")
        return

    # --- Phase 3: Download completed files from Google Drive ---
    print()
    print("Downloading completed files from Google Drive...")

    drive_service = _build_drive_service()

    if drive_service is None:
        print()
        print("=" * 60)
        print("Drive download unavailable. Tasks completed on GEE.")
        print(f"Files are in your Google Drive folder: {folder}/")
        print()
        print("Download manually or re-authenticate with Drive scope:")
        print(
            "  earthengine authenticate"
            " --scopes=https://www.googleapis.com/auth/earthengine"
            ",https://www.googleapis.com/auth/drive"
            ",https://www.googleapis.com/auth/cloud-platform"
            ",https://www.googleapis.com/auth/devstorage.full_control"
        )
        print()
        print("Expected files:")
        for i in sorted(completed):
            export = tasks[i]["export"]
            print(f"  {export['filename']}.tif -> {export['output_path']}")
        print("=" * 60)
        return

    downloaded = 0
    download_errors = 0

    for i in sorted(completed):
        export = tasks[i]["export"]
        filename = export["filename"]
        output_path = export["output_path"]
        desc = export["description"]

        print(f"\n  Downloading: {desc}")

        # Search in the target folder first, then at root
        file_id = _find_drive_file(drive_service, filename, folder_name=folder)
        if file_id is None:
            file_id = _find_drive_file(drive_service, filename)

        if file_id is None:
            print(f"    ERROR: Could not find '{filename}' on Google Drive.")
            print(f"    Check Drive folder '{folder}/' manually.")
            download_errors += 1
            continue

        try:
            _download_drive_file(drive_service, file_id, output_path)
            downloaded += 1
            _delete_drive_file(drive_service, file_id, filename)
        except Exception as exc:
            print(f"    ERROR downloading {filename}: {exc}")
            download_errors += 1

    print()
    print(f"Download summary: {downloaded} downloaded, {download_errors} errors")

    if failed:
        print("\nFailed export tasks:")
        for i, error_msg in failed.items():
            print(f"  {tasks[i]['export']['description']}: {error_msg}")

    if downloaded == len(completed):
        print(f"\nAll {downloaded} files downloaded successfully ({total_elapsed})")
