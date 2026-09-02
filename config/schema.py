"""Shared dataset schema for Instinct+.

Every other module imports its column names, file patterns and collection
filters from here instead of hardcoding strings, so the title pair, the
thumbnail pair and the interface pair all read the same tables.

Changes to this file require agreement from all three pairs.
"""

# --- column names -----------------------------------------------------------

VIDEO_ID = "video_id"
TITLE = "title"
CHANNEL_ID = "channel_id"
SUBSCRIBER_COUNT = "subscriber_count"
VIEW_COUNT = "view_count"
PUBLISHED_AT = "published_at"
CONTENT_TYPE = "content_type"
THUMBNAIL_FILE = "thumbnail_file"

COLUMNS = [
    VIDEO_ID,
    TITLE,
    CHANNEL_ID,
    SUBSCRIBER_COUNT,
    VIEW_COUNT,
    PUBLISHED_AT,
    CONTENT_TYPE,
    THUMBNAIL_FILE,
]

# --- files ------------------------------------------------------------------

# naming for images in data/thumbnails/
THUMBNAIL_PATTERN = "{video_id}.jpg"

# --- content types ----------------------------------------------------------

# placeholder. the team has not finalised this list.
CONTENT_TYPES = ["gaming", "cooking", "finance", "education"]

# --- collection filters -----------------------------------------------------

# the API rounds subscriber counts to three significant figures, so below this
# the channel size feature is too coarse to be useful
MIN_SUBSCRIBERS = 10_000

# view counts need time to settle before they represent the video's reach
MIN_DAYS_SINCE_PUBLISH = 30

# older videos accumulate views for longer, which confounds the target variable
MAX_DAYS_SINCE_PUBLISH = 365
