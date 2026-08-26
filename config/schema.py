"""Shared schema. Changing anything here requires agreement from all three pairs."""

# Column names in the collected dataset
VIDEO_ID = "video_id"
TITLE = "title"
CHANNEL_ID = "channel_id"
SUBSCRIBER_COUNT = "subscriber_count"
VIEW_COUNT = "view_count"
PUBLISHED_AT = "published_at"
CONTENT_TYPE = "content_type"
THUMBNAIL_FILE = "thumbnail_file"

COLUMNS = [
    VIDEO_ID, TITLE, CHANNEL_ID, SUBSCRIBER_COUNT,
    VIEW_COUNT, PUBLISHED_AT, CONTENT_TYPE, THUMBNAIL_FILE,
]

# Thumbnail files are named "{video_id}.jpg" in data/thumbnails/
THUMBNAIL_PATTERN = "{video_id}.jpg"

CONTENT_TYPES = ["gaming", "cooking", "finance", "education"]

# Collection filters
MIN_SUBSCRIBERS = 10_000       # API rounds to 3 sig figs below this
MIN_DAYS_SINCE_PUBLISH = 30    # let view counts settle
