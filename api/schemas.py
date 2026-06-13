"""
Pydantic request/response models for both prediction endpoints.

ArticleFeatures        -> input for POST /predict/article-shares
ArticleSharesResponse  -> output for POST /predict/article-shares

PersonFeatures         -> input for POST /predict/income-bracket
IncomeBracketResponse  -> output for POST /predict/income-bracket
"""

from enum import Enum
from pydantic import BaseModel, Field


# ============================================================
# Regression: Online News Popularity
# ============================================================

class ArticleFeatures(BaseModel):
    """
    All 58 content/metadata features used by the regression model,
    in the exact order the model was trained on. Defaults represent
    a "typical" article so this schema can be submitted as-is for a
    quick test, then overridden field-by-field for real predictions.
    """
    n_tokens_title: float = Field(10.0, description="Number of words in the title")
    n_tokens_content: float = Field(500.0, description="Number of words in the content")
    n_unique_tokens: float = Field(0.55, description="Rate of unique words in the content")
    n_non_stop_words: float = Field(1.0, description="Rate of non-stop words in the content")
    n_non_stop_unique_tokens: float = Field(0.7, description="Rate of unique non-stop words in the content")
    num_hrefs: float = Field(10.0, description="Number of links")
    num_self_hrefs: float = Field(3.0, description="Number of links to other Mashable articles")
    num_imgs: float = Field(4.0, description="Number of images")
    num_videos: float = Field(1.0, description="Number of videos")
    average_token_length: float = Field(4.5, description="Average length of the words in the content")
    num_keywords: float = Field(7.0, description="Number of keywords in the article metadata")
    data_channel_is_lifestyle: int = Field(0, ge=0, le=1, description="Is the article in the Lifestyle channel? (0 or 1)")
    data_channel_is_entertainment: int = Field(0, ge=0, le=1, description="Is the article in the Entertainment channel? (0 or 1)")
    data_channel_is_bus: int = Field(0, ge=0, le=1, description="Is the article in the Business channel? (0 or 1)")
    data_channel_is_socmed: int = Field(0, ge=0, le=1, description="Is the article in the Social Media channel? (0 or 1)")
    data_channel_is_tech: int = Field(1, ge=0, le=1, description="Is the article in the Tech channel? (0 or 1)")
    data_channel_is_world: int = Field(0, ge=0, le=1, description="Is the article in the World channel? (0 or 1)")
    kw_min_min: float = Field(-1.0, description="Worst keyword: minimum shares it has ever earned")
    kw_max_min: float = Field(500.0, description="Worst keyword: maximum shares it has ever earned")
    kw_avg_min: float = Field(200.0, description="Worst keyword: average shares it earns")
    kw_min_max: float = Field(0.0, description="Best keyword: minimum shares it has ever earned")
    kw_max_max: float = Field(700000.0, description="Best keyword: maximum shares it has ever earned")
    kw_avg_max: float = Field(250000.0, description="Best keyword: average shares it earns")
    kw_min_avg: float = Field(1000.0, description="Average keyword: minimum shares it has ever earned")
    kw_max_avg: float = Field(5000.0, description="Average keyword: maximum shares it has ever earned")
    kw_avg_avg: float = Field(3000.0, description="Average keyword: average shares it earns")
    self_reference_min_shares: float = Field(1000.0, description="Min. shares of articles referenced/linked by this one")
    self_reference_max_shares: float = Field(5000.0, description="Max. shares of articles referenced/linked by this one")
    self_reference_avg_sharess: float = Field(3000.0, description="Avg. shares of articles referenced/linked by this one")
    weekday_is_monday: int = Field(0, ge=0, le=1, description="Was the article published on a Monday? (0 or 1)")
    weekday_is_tuesday: int = Field(1, ge=0, le=1, description="Was the article published on a Tuesday? (0 or 1)")
    weekday_is_wednesday: int = Field(0, ge=0, le=1, description="Was the article published on a Wednesday? (0 or 1)")
    weekday_is_thursday: int = Field(0, ge=0, le=1, description="Was the article published on a Thursday? (0 or 1)")
    weekday_is_friday: int = Field(0, ge=0, le=1, description="Was the article published on a Friday? (0 or 1)")
    weekday_is_saturday: int = Field(0, ge=0, le=1, description="Was the article published on a Saturday? (0 or 1)")
    weekday_is_sunday: int = Field(0, ge=0, le=1, description="Was the article published on a Sunday? (0 or 1)")
    is_weekend: int = Field(0, ge=0, le=1, description="Was the article published on a weekend? (0 or 1)")
    LDA_00: float = Field(0.2, description="Closeness to LDA topic 0 (auto-discovered topic cluster)")
    LDA_01: float = Field(0.2, description="Closeness to LDA topic 1 (auto-discovered topic cluster)")
    LDA_02: float = Field(0.2, description="Closeness to LDA topic 2 (auto-discovered topic cluster)")
    LDA_03: float = Field(0.2, description="Closeness to LDA topic 3 (auto-discovered topic cluster)")
    LDA_04: float = Field(0.2, description="Closeness to LDA topic 4 (auto-discovered topic cluster)")
    global_subjectivity: float = Field(0.45, description="Overall text subjectivity (0=objective, 1=subjective)")
    global_sentiment_polarity: float = Field(0.1, description="Overall text sentiment polarity (negative to positive)")
    global_rate_positive_words: float = Field(0.04, description="Rate of positive words in the content")
    global_rate_negative_words: float = Field(0.02, description="Rate of negative words in the content")
    rate_positive_words: float = Field(0.7, description="Rate of positive words among non-neutral words")
    rate_negative_words: float = Field(0.3, description="Rate of negative words among non-neutral words")
    avg_positive_polarity: float = Field(0.35, description="Average polarity of positive words")
    min_positive_polarity: float = Field(0.05, description="Minimum polarity of positive words")
    max_positive_polarity: float = Field(0.7, description="Maximum polarity of positive words")
    avg_negative_polarity: float = Field(-0.25, description="Average polarity of negative words")
    min_negative_polarity: float = Field(-0.5, description="Minimum polarity of negative words")
    max_negative_polarity: float = Field(-0.1, description="Maximum polarity of negative words")
    title_subjectivity: float = Field(0.3, description="Title subjectivity (0=objective, 1=subjective)")
    title_sentiment_polarity: float = Field(0.1, description="Title sentiment polarity (negative to positive)")
    abs_title_subjectivity: float = Field(0.2, description="Absolute subjectivity level of the title")
    abs_title_sentiment_polarity: float = Field(0.1, description="Absolute sentiment polarity level of the title")


class ArticleSharesResponse(BaseModel):
    predicted_shares: float = Field(..., description="Predicted number of social media shares")
    predicted_log_shares: float = Field(..., description="Predicted value on the log(shares + 1) scale the model was trained on")
    model_used: str = Field(..., description="Name of the underlying regression model")


# ============================================================
# Classification: Adult / Census Income
# ============================================================

class WorkclassEnum(str, Enum):
    private = "Private"
    self_emp_not_inc = "Self-emp-not-inc"
    self_emp_inc = "Self-emp-inc"
    federal_gov = "Federal-gov"
    local_gov = "Local-gov"
    state_gov = "State-gov"
    without_pay = "Without-pay"
    never_worked = "Never-worked"


class EducationEnum(str, Enum):
    bachelors = "Bachelors"
    some_college = "Some-college"
    eleventh = "11th"
    hs_grad = "HS-grad"
    prof_school = "Prof-school"
    assoc_acdm = "Assoc-acdm"
    assoc_voc = "Assoc-voc"
    ninth = "9th"
    seventh_eighth = "7th-8th"
    twelfth = "12th"
    masters = "Masters"
    first_fourth = "1st-4th"
    tenth = "10th"
    doctorate = "Doctorate"
    fifth_sixth = "5th-6th"
    preschool = "Preschool"


class MaritalStatusEnum(str, Enum):
    married_civ_spouse = "Married-civ-spouse"
    divorced = "Divorced"
    never_married = "Never-married"
    separated = "Separated"
    widowed = "Widowed"
    married_spouse_absent = "Married-spouse-absent"
    married_af_spouse = "Married-AF-spouse"


class OccupationEnum(str, Enum):
    tech_support = "Tech-support"
    craft_repair = "Craft-repair"
    other_service = "Other-service"
    sales = "Sales"
    exec_managerial = "Exec-managerial"
    prof_specialty = "Prof-specialty"
    handlers_cleaners = "Handlers-cleaners"
    machine_op_inspct = "Machine-op-inspct"
    adm_clerical = "Adm-clerical"
    farming_fishing = "Farming-fishing"
    transport_moving = "Transport-moving"
    priv_house_serv = "Priv-house-serv"
    protective_serv = "Protective-serv"
    armed_forces = "Armed-Forces"


class RelationshipEnum(str, Enum):
    wife = "Wife"
    own_child = "Own-child"
    husband = "Husband"
    not_in_family = "Not-in-family"
    other_relative = "Other-relative"
    unmarried = "Unmarried"


class RaceEnum(str, Enum):
    white = "White"
    asian_pac_islander = "Asian-Pac-Islander"
    amer_indian_eskimo = "Amer-Indian-Eskimo"
    other = "Other"
    black = "Black"


class SexEnum(str, Enum):
    female = "Female"
    male = "Male"


class PersonFeatures(BaseModel):
    """
    The 12 demographic/employment features used by the classification
    pipeline. Categorical fields use enums covering the categories present
    in the training data; 'native_country' is left as free text since the
    pipeline's OneHotEncoder(handle_unknown='ignore') safely handles any
    country not seen during training by encoding it as all-zeros.
    """
    age: int = Field(38, ge=17, le=90, description="Age in years")
    workclass: WorkclassEnum = Field(WorkclassEnum.private, description="Type of employer")
    education: EducationEnum = Field(EducationEnum.hs_grad, description="Highest level of education completed")
    marital_status: MaritalStatusEnum = Field(MaritalStatusEnum.married_civ_spouse, description="Marital status")
    occupation: OccupationEnum = Field(OccupationEnum.prof_specialty, description="Type of job")
    relationship: RelationshipEnum = Field(RelationshipEnum.husband, description="Relationship role within household")
    race: RaceEnum = Field(RaceEnum.white, description="Race")
    sex: SexEnum = Field(SexEnum.male, description="Sex")
    capital_gain: float = Field(0.0, ge=0, description="Income from investments/capital gains (USD)")
    capital_loss: float = Field(0.0, ge=0, description="Losses from investments/capital losses (USD)")
    hours_per_week: int = Field(40, ge=1, le=99, description="Number of hours worked per week")
    native_country: str = Field("United-States", description="Country of origin")


class IncomeBracketResponse(BaseModel):
    predicted_income: str = Field(..., description="'<=50K' or '>50K'")
    probability_over_50k: float = Field(..., description="Predicted probability that income exceeds $50K")
    model_used: str = Field(..., description="Name of the underlying classification model (a full preprocessing + model pipeline)")
