import os
import tempfile


DISABLED_VALUES = ("0", "false", "no", "off")


class CFGVerificationError(RuntimeError):
    """Raised when CFG extraction or comparison cannot be completed."""


def is_cfg_verification_enabled():
    """Return whether CFG verification should be applied to successful samples."""
    value = os.environ.get("RLAEG_ENABLE_CFG_CHECK", "1").strip().lower()
    return value not in DISABLED_VALUES


def extract_cfg_signature(file_path):
    """Extract a comparable CFG signature from a PE file path."""
    try:
        import angr
    except ImportError as exc:
        raise CFGVerificationError(
            "angr is required for CFG verification. Install requirements.txt "
            "or set RLAEG_ENABLE_CFG_CHECK=0 to disable this check."
        ) from exc

    try:
        project = angr.Project(str(file_path), load_options={"auto_load_libs": False})
        cfg = project.analyses.CFG()
        functions = dict(cfg.kb.functions)
    except Exception as exc:
        raise CFGVerificationError(
            "failed to extract CFG from {}".format(file_path)
        ) from exc

    return {address: str(function) for address, function in functions.items()}


def extract_cfg_signature_from_bytes(bytez, file_name=None):
    """Write bytes to a temporary PE file and extract its CFG signature."""
    suffix = ".exe"
    if file_name:
        _, extension = os.path.splitext(file_name)
        if extension:
            suffix = extension

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(bytez)
            tmp_path = tmp_file.name
        return extract_cfg_signature(tmp_path)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def compare_cfg_signatures(original_signature, modified_signature):
    """Compare two CFG signatures and return a compact comparison summary."""
    original_addresses = set(original_signature.keys())
    modified_addresses = set(modified_signature.keys())
    shared_addresses = original_addresses & modified_addresses
    changed_addresses = [
        address for address in shared_addresses
        if original_signature[address] != modified_signature[address]
    ]
    missing_addresses = original_addresses - modified_addresses
    added_addresses = modified_addresses - original_addresses

    same = (
        not missing_addresses and
        not added_addresses and
        not changed_addresses
    )
    return {
        "same": same,
        "original_function_count": len(original_signature),
        "modified_function_count": len(modified_signature),
        "missing_count": len(missing_addresses),
        "added_count": len(added_addresses),
        "changed_count": len(changed_addresses),
    }


def verify_modified_cfg(original_signature, modified_bytez, file_name=None):
    """Extract and compare the modified sample CFG against the original CFG."""
    if original_signature is None:
        raise CFGVerificationError("original CFG signature is missing")

    modified_signature = extract_cfg_signature_from_bytes(modified_bytez, file_name)
    return compare_cfg_signatures(original_signature, modified_signature)
