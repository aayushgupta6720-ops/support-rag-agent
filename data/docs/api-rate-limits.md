# API Rate Limits

The API enforces a limit of 100 requests per minute per API key on the free
tier, and 1,000 requests per minute on paid plans. Limits are applied using a
sliding window, not a fixed per-minute bucket.

When you exceed your limit, the API returns HTTP 429 with a `Retry-After`
header indicating how many seconds to wait before retrying. We recommend
exponential backoff with jitter rather than retrying immediately.

Rate limit usage is tracked per API key, not per account, so creating
additional keys does not increase your effective limit — all keys on an
account share the same quota. If you need a higher limit for a legitimate
use case, contact support with your expected request volume.

Bulk export endpoints have a separate, lower limit of 10 requests per minute
due to their higher processing cost.
