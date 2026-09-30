# Managing API Keys

Create API keys under Account > API Keys. Send a key with each request in the
`Authorization: Bearer <key>` header.

A new key is shown only once, when you create it, so copy it somewhere safe
straight away. If you lose a key, create a new one and revoke the old one; we
can't show an existing key again.

Each key is either read-only or read-write. Use read-only keys for
integrations that only need to fetch data.

Revoking a key takes effect immediately: requests that use it get HTTP 401.
To rotate a key without downtime, create the new key, switch your clients
over to it, and then revoke the old one.

Keys belong to your account, not to the person who created them, so they keep
working if that person leaves a team account. Usage from every key counts
against your account; see API Rate Limits.
