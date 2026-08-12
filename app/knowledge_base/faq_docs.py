"""
Example FAQ / help-doc content for a fictitious product, "Acme Cloud Storage".
In a real deployment this would be replaced with actual help-center articles,
pulled from a CMS, Zendesk/Confluence export, etc.

Each entry becomes one or more chunks in the vector store.
"""

FAQ_DOCS = [
    {
        "doc_id": "kb-001",
        "title": "Resetting your password",
        "text": (
            "To reset your Acme Cloud Storage password, go to the login page and "
            "click 'Forgot password'. Enter the email address on your account and "
            "we will send a reset link that is valid for 30 minutes. If you do not "
            "receive the email within a few minutes, check your spam folder. "
            "For security, password reset links cannot be resent more than 5 times "
            "per hour for the same account."
        ),
    },
    {
        "doc_id": "kb-002",
        "title": "Storage plan limits and overages",
        "text": (
            "The Free plan includes 5GB of storage. The Pro plan includes 500GB for "
            "$9.99/month. The Business plan includes 5TB for $29.99/month per user. "
            "If you exceed your plan's storage limit, uploads are paused but you will "
            "not lose existing files. You can free up space, delete files, or upgrade "
            "your plan at any time from Account Settings > Billing > Change Plan."
        ),
    },
    {
        "doc_id": "kb-003",
        "title": "Requesting a refund",
        "text": (
            "Acme Cloud Storage offers a 14-day money-back guarantee on all new paid "
            "subscriptions. To request a refund, go to Account Settings > Billing > "
            "Billing History and click 'Request refund' next to the relevant charge, "
            "or email billing@acmecloud.example. Refunds are processed within 5-7 "
            "business days back to the original payment method. Refunds are not "
            "available for renewals after the first 14 days of a billing cycle."
        ),
    },
    {
        "doc_id": "kb-004",
        "title": "Sharing files and folders",
        "text": (
            "You can share a file or folder by right-clicking it and selecting "
            "'Share'. You can invite people by email (they need an Acme account to "
            "edit, but not to view) or generate a public link. Shared links can be "
            "set to 'view only' or 'can edit', and can optionally require a password "
            "or expire after a set number of days."
        ),
    },
    {
        "doc_id": "kb-005",
        "title": "Two-factor authentication (2FA) setup",
        "text": (
            "To enable two-factor authentication, go to Account Settings > Security > "
            "Two-Factor Authentication and click 'Enable'. You can use an authenticator "
            "app (Google Authenticator, Authy) via QR code, or receive codes by SMS. "
            "We strongly recommend saving the provided backup codes in a safe place, "
            "since they are the only way to regain access if you lose your 2FA device."
        ),
    },
    {
        "doc_id": "kb-006",
        "title": "Recovering deleted files",
        "text": (
            "Deleted files move to the Trash and are kept there for 30 days before "
            "being permanently removed. To recover a file, open Trash, select the "
            "file, and click 'Restore'. Business plan accounts have an extended "
            "90-day retention window and can request permanent-deletion recovery "
            "from support within 7 days of permanent deletion, on a best-effort basis."
        ),
    },
    {
        "doc_id": "kb-007",
        "title": "Cancelling your subscription",
        "text": (
            "To cancel, go to Account Settings > Billing > Manage Subscription and "
            "click 'Cancel Plan'. Your paid features remain active until the end of "
            "the current billing period, after which your account reverts to the "
            "Free plan. If your storage usage exceeds the Free plan limit at that "
            "point, uploads will be paused until you delete files or resubscribe."
        ),
    },
    {
        "doc_id": "kb-008",
        "title": "Desktop sync client troubleshooting",
        "text": (
            "If the desktop sync client shows files stuck 'Syncing' for more than a "
            "few minutes: 1) check your internet connection, 2) confirm you are not "
            "over your storage limit, 3) restart the sync client from the tray icon "
            "menu, 4) ensure the app has disk access permission in your OS privacy "
            "settings. If the issue persists, generate a diagnostic log via "
            "Settings > Help > Export Diagnostics and attach it to a support ticket."
        ),
    },
]
