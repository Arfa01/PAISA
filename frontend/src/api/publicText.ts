const NETWORK_URL_PATTERN = /\b(?:https?|wss?|ftp):\/\/[^\s"'<>]+/gi;

const LOCAL_PATH_PATTERNS = [
  /(?:^|[^A-Za-z0-9_])file:/i,
  /(?:^|[^A-Za-z0-9_/])\/{1,2}(?=\S)/,
  /(?:^|[^A-Za-z0-9_\\])\\\\(?=\S)/,
  /(?:^|[^A-Za-z0-9_])[A-Za-z]:[\\/]/,
];

const EXECUTABLE_MODEL_SUFFIX =
  /\.(?:joblib|pkl|pickle|keras|h5|hdf5|onnx|pt|pth|pb|ckpt|safetensors)(?=$|[^A-Za-z0-9])/i;

function containsProhibitedUrlMetadata(value: string): boolean {
  for (const match of value.matchAll(NETWORK_URL_PATTERN)) {
    try {
      const url = new URL(match[0]);
      const metadata = `${decodeURIComponent(url.search)} ${decodeURIComponent(url.hash)}`;
      const decodedPath = decodeURIComponent(url.pathname);
      if (
        LOCAL_PATH_PATTERNS.some((pattern) => pattern.test(metadata)) ||
        EXECUTABLE_MODEL_SUFFIX.test(metadata) ||
        EXECUTABLE_MODEL_SUFFIX.test(decodedPath)
      ) {
        return true;
      }
    } catch {
      return true;
    }
  }
  return false;
}

function stringContainsProhibitedPublicArtifactText(value: string): boolean {
  const withoutWebUrls = value.replace(NETWORK_URL_PATTERN, "");
  return (
    containsProhibitedUrlMetadata(value) ||
    LOCAL_PATH_PATTERNS.some((pattern) => pattern.test(withoutWebUrls)) ||
    EXECUTABLE_MODEL_SUFFIX.test(value)
  );
}

export function containsProhibitedPublicArtifactText(value: unknown): boolean {
  if (typeof value === "string") {
    return stringContainsProhibitedPublicArtifactText(value);
  }
  if (Array.isArray(value)) {
    return value.some(containsProhibitedPublicArtifactText);
  }
  if (value && typeof value === "object") {
    return Object.entries(value).some(
      ([key, nested]) =>
        stringContainsProhibitedPublicArtifactText(key) ||
        containsProhibitedPublicArtifactText(nested),
    );
  }
  return false;
}
