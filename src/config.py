import os
import pycountry
from fuzzywuzzy import process

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

OFFLINE = os.getenv("LOGISTICS_OFFLINE", "1").strip().lower() not in {"0", "false", "no"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, 'DataCoSupplyChainDataset.csv')
ENRICHED_OUTPUT_PATH = os.path.join(DATA_DIR, 'enriched_dataset.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_profit_classifier.joblib')
FEATURE_META_PATH = os.path.join(MODELS_DIR, 'feature_meta.joblib')

GEOCODING_USER_AGENT = "CourseworkGeocoder/1.0"

CATEGORICAL_COLUMNS_USED = [
    'Type', 'Shipping Mode', 'Category Name', 'Order Region',
    'Order City', 'Order Country', 'Market', 'Customer Segment', 'Department Name'
]



SPANISH_TO_ENGLISH = {
    'Francia':                          'France',
    'Alemania':                         'Germany',
    'Reino Unido':                      'UK',
    'Brasil':                           'Brazil',
    'Japón':                            'Japan',
    'Países Bajos':                     'Netherlands',
    'Turquía':                          'Turkey',
    'Marruecos':                        'Morocco',
    'Corea del Sur':                    'South Korea',
    'Filipinas':                        'Philippines',
    'República Dominicana':             'Dominican Republic',
    'Costa de Marfil':                  'Ivory Coast',
    'Argelia':                          'Algeria',
    'Egipto':                           'Egypt',
    'España':                           'Spain',
    'México':                           'Mexico',
    'Italia':                           'Italy',
    'Rusia':                            'Russia',
    'Singapur':                         'Singapore',
    'Nueva Zelanda':                    'New Zealand',
    'Suecia':                           'Sweden',
    'Noruega':                          'Norway',
    'Polonia':                          'Poland',
    'Rumania':                          'Romania',
    'Bélgica':                          'Belgium',
    'Irlanda':                          'Ireland',
    'Finlandia':                        'Finland',
    'Pakistán':                         'Pakistan',
    'Bangladés':                        'Bangladesh',
    'Tailandia':                        'Thailand',
    'Malasia':                          'Malaysia',
    'Arabia Saudí':                     'Saudi Arabia',
    'Irán':                             'Iran',
    'Irak':                             'Iraq',
    'Kazajistán':                       'Kazakhstan',
    'Uzbekistán':                       'Uzbekistan',
    'Kirguistán':                       'Kyrgyzstan',
    'Ucrania':                          'Ukraine',
    'Eslovaquia':                       'Slovakia',
    'Países Bajos':                     'Netherlands',
    'Panamá':                           'Panama',
    'Perú':                             'Peru',
    'Camerún':                          'Cameroon',
    'Etiopía':                          'Ethiopia',
    'Benín':                            'Benin',
    'Níger':                            'Niger',
    'República Democrática del Congo':  'Democratic Republic of the Congo',
    'SudAfrica':                        'South Africa',
    'Afganistán':                       'Afghanistan',
    'Myanmar (Birmania)':               'Myanmar',
    'Kenia':                            'Kenya',
    'Ruanda':                           'Rwanda',
    'Sudán':                            'Sudan',
    'Togo':                             'Togo',
    'Nepal':                            'Nepal',
    'Yemen':                            'Yemen',
    'Trinidad y Tobago':                'Trinidad and Tobago',
    'Venezuela':                        'Venezuela',
    'Colombia':                         'Colombia',
    'Ecuador':                          'Ecuador',
    'Bolivia':                          'Bolivia',
    'Uruguay':                          'Uruguay',
    'Honduras':                         'Honduras',
    'Guatemala':                        'Guatemala',
    'El Salvador':                      'El Salvador',
    'Nicaragua':                        'Nicaragua',
    'Cuba':                             'Cuba',
    'Estados Unidos':                   'USA',
    'Canada':                           'Canada',
    'Australia':                        'Australia',
    'China':                            'China',
    'India':                            'India',
    'Indonesia':                        'Indonesia',
    'Nigeria':                          'Nigeria',
    'Argentina':                        'Argentina',
    'Chile':                            'Chile',
    'Portugal':                         'Portugal',
    'Austria':                          'Austria',
    'Georgia':                          'Georgia',
    'Bulgaria':                         'Bulgaria',
    'Israel':                           'Israel',
    'Mongolia':                         'Mongolia',
    'Vietnam':                          'Vietnam',
    'Angola':                           'Angola',
    'Senegal':                          'Senegal',
    'Tanzania':                         'Tanzania',
    'Ghana':                            'Ghana',
    'Somalia':                          'Somalia',
    'Liberia':                          'Liberia',
    'Zambia':                           'Zambia',
    'Jamaica':                          'Jamaica',
    'Madagascar':                       'Madagascar',
    'Mozambique':                       'Mozambique',
    'Guinea':                           'Guinea',
}

QUICK_FIX = {
    'usa': 'USA', 'united states': 'USA', 'united states of america': 'USA',
    'u.s.': 'USA', 'u.s.a.': 'USA', 'us': 'USA',
    'uk': 'UK', 'united kingdom': 'UK',
}

FUZZY_MATCH_CANONICAL_SCORE = 92
FUZZY_MATCH_FALLBACK_SCORE = 96

CANONICAL_NAMES = {
    'United States': 'USA',
    'United Kingdom': 'UK',
    'Korea, Republic of': 'South Korea',
    'Congo, The Democratic Republic of the': 'Democratic Republic of the Congo',
    "Côte d'Ivoire": 'Ivory Coast',
}

CANONICAL_COUNTRY_BY_LOWER = {
    country.name.lower(): country.name for country in pycountry.countries
}


def normalize_country_name(raw_name: str) -> str:
    if not raw_name or not isinstance(raw_name, str):
        return 'Unknown'

    cleaned_name = raw_name.strip()
    if not cleaned_name:
        return 'Unknown'
    mapped = _normalize_direct(cleaned_name)
    if mapped:
        return mapped

    name_lower = cleaned_name.lower()
    mapped = _normalize_quick(name_lower)
    if mapped:
        return mapped

    mapped = _normalize_canonical(name_lower)
    if mapped:
        return mapped

    mapped = _normalize_fuzzy(name_lower)
    if mapped:
        return mapped

    return 'Unknown'


def _normalize_direct(cleaned_name: str) -> str | None:
    return SPANISH_TO_ENGLISH.get(cleaned_name)


def _normalize_quick(name_lower: str) -> str | None:
    return QUICK_FIX.get(name_lower)


def _normalize_canonical(name_lower: str) -> str | None:
    canonical = CANONICAL_COUNTRY_BY_LOWER.get(name_lower)
    if canonical:
        return CANONICAL_NAMES.get(canonical, canonical)
    try:
        country = pycountry.countries.search_fuzzy(name_lower)[0]
        return CANONICAL_NAMES.get(country.name, country.name)
    except LookupError:
        return None


def _normalize_fuzzy(name_lower: str) -> str | None:
    best_match, score = process.extractOne(name_lower, list(CANONICAL_COUNTRY_BY_LOWER))
    if score >= FUZZY_MATCH_CANONICAL_SCORE:
        canonical = CANONICAL_COUNTRY_BY_LOWER[best_match]
        return CANONICAL_NAMES.get(canonical, canonical)
    if score >= FUZZY_MATCH_FALLBACK_SCORE:
        return best_match.title()
    return None


COUNTRY_STATS_2018 = {
    'USA':                {'gdp_ppp': 62800,  'lpi': 3.89, 'inf': 2.4,  'tariff_zone': 'NAFTA'},
    'France':             {'gdp_ppp': 45300,  'lpi': 3.84, 'inf': 1.8,  'tariff_zone': 'EU'},
    'Germany':            {'gdp_ppp': 52000,  'lpi': 4.20, 'inf': 1.7,  'tariff_zone': 'EU'},
    'UK':                 {'gdp_ppp': 43000,  'lpi': 3.99, 'inf': 2.5,  'tariff_zone': 'EU'},
    'Italy':              {'gdp_ppp': 40000,  'lpi': 3.70, 'inf': 1.2,  'tariff_zone': 'EU'},
    'Spain':              {'gdp_ppp': 41000,  'lpi': 3.83, 'inf': 1.7,  'tariff_zone': 'EU'},
    'Netherlands':        {'gdp_ppp': 61000,  'lpi': 4.05, 'inf': 1.8,  'tariff_zone': 'EU'},
    'Belgium':            {'gdp_ppp': 51000,  'lpi': 3.66, 'inf': 2.3,  'tariff_zone': 'EU'},
    'Sweden':             {'gdp_ppp': 55000,  'lpi': 3.76, 'inf': 2.0,  'tariff_zone': 'EU'},
    'Norway':             {'gdp_ppp': 78000,  'lpi': 3.93, 'inf': 2.7,  'tariff_zone': 'WTO'},
    'Poland':             {'gdp_ppp': 33000,  'lpi': 3.50, 'inf': 1.6,  'tariff_zone': 'EU'},
    'Romania':            {'gdp_ppp': 28000,  'lpi': 3.15, 'inf': 4.6,  'tariff_zone': 'EU'},
    'Austria':            {'gdp_ppp': 56000,  'lpi': 3.84, 'inf': 2.1,  'tariff_zone': 'EU'},
    'Ireland':            {'gdp_ppp': 78000,  'lpi': 3.79, 'inf': 0.7,  'tariff_zone': 'EU'},
    'Finland':            {'gdp_ppp': 49000,  'lpi': 3.97, 'inf': 1.2,  'tariff_zone': 'EU'},
    'Portugal':           {'gdp_ppp': 35000,  'lpi': 3.64, 'inf': 1.4,  'tariff_zone': 'EU'},
    'Slovakia':           {'gdp_ppp': 33000,  'lpi': 3.32, 'inf': 2.5,  'tariff_zone': 'EU'},
    'Bulgaria':           {'gdp_ppp': 22000,  'lpi': 2.84, 'inf': 2.8,  'tariff_zone': 'EU'},

    'Canada':             {'gdp_ppp': 52000,  'lpi': 3.92, 'inf': 2.3,  'tariff_zone': 'NAFTA'},
    'Mexico':             {'gdp_ppp': 20000,  'lpi': 3.05, 'inf': 4.9,  'tariff_zone': 'NAFTA'},

    'Brazil':             {'gdp_ppp': 15000,  'lpi': 2.99, 'inf': 3.7,  'tariff_zone': 'LATAM'},
    'Argentina':          {'gdp_ppp': 20000,  'lpi': 2.89, 'inf': 25.0, 'tariff_zone': 'LATAM'},
    'Colombia':           {'gdp_ppp': 14000,  'lpi': 2.94, 'inf': 3.2,  'tariff_zone': 'LATAM'},
    'Chile':              {'gdp_ppp': 25000,  'lpi': 3.32, 'inf': 2.4,  'tariff_zone': 'LATAM'},
    'Peru':               {'gdp_ppp': 14000,  'lpi': 2.70, 'inf': 1.3,  'tariff_zone': 'LATAM'},
    'Venezuela':          {'gdp_ppp': 12000,  'lpi': 2.30, 'inf': 65536,'tariff_zone': 'LATAM'},
    'Ecuador':            {'gdp_ppp': 11000,  'lpi': 2.78, 'inf': 0.3,  'tariff_zone': 'LATAM'},
    'Bolivia':            {'gdp_ppp': 8000,   'lpi': 2.55, 'inf': 2.8,  'tariff_zone': 'LATAM'},
    'Uruguay':            {'gdp_ppp': 22000,  'lpi': 2.88, 'inf': 7.9,  'tariff_zone': 'LATAM'},
    'Panama':             {'gdp_ppp': 25000,  'lpi': 3.28, 'inf': 0.8,  'tariff_zone': 'LATAM'},
    'Honduras':           {'gdp_ppp': 5500,   'lpi': 2.60, 'inf': 4.3,  'tariff_zone': 'LATAM'},
    'Guatemala':          {'gdp_ppp': 8500,   'lpi': 2.71, 'inf': 3.8,  'tariff_zone': 'LATAM'},
    'El Salvador':        {'gdp_ppp': 9000,   'lpi': 2.60, 'inf': 1.1,  'tariff_zone': 'LATAM'},
    'Nicaragua':          {'gdp_ppp': 5500,   'lpi': 2.49, 'inf': 4.9,  'tariff_zone': 'LATAM'},
    'Cuba':               {'gdp_ppp': 8000,   'lpi': 2.30, 'inf': 5.5,  'tariff_zone': 'LATAM'},
    'Dominican Republic': {'gdp_ppp': 18000,  'lpi': 2.80, 'inf': 3.5,  'tariff_zone': 'LATAM'},
    'Jamaica':            {'gdp_ppp': 10000,  'lpi': 2.60, 'inf': 3.7,  'tariff_zone': 'LATAM'},
    'Trinidad and Tobago':{'gdp_ppp': 22000,  'lpi': 2.58, 'inf': 1.0,  'tariff_zone': 'LATAM'},

    'China':              {'gdp_ppp': 16000,  'lpi': 3.61, 'inf': 2.1,  'tariff_zone': 'WTO'},
    'Japan':              {'gdp_ppp': 48000,  'lpi': 4.03, 'inf': 0.5,  'tariff_zone': 'WTO'},
    'South Korea':        {'gdp_ppp': 45000,  'lpi': 3.88, 'inf': 1.5,  'tariff_zone': 'WTO'},
    'Australia':          {'gdp_ppp': 50000,  'lpi': 3.75, 'inf': 1.9,  'tariff_zone': 'WTO'},
    'New Zealand':        {'gdp_ppp': 42000,  'lpi': 3.88, 'inf': 1.9,  'tariff_zone': 'WTO'},
    'Russia':             {'gdp_ppp': 28000,  'lpi': 2.96, 'inf': 4.3,  'tariff_zone': 'WTO'},
    'Turkey':             {'gdp_ppp': 28000,  'lpi': 3.15, 'inf': 16.3, 'tariff_zone': 'WTO'},
    'Saudi Arabia':       {'gdp_ppp': 55000,  'lpi': 3.16, 'inf': 2.5,  'tariff_zone': 'WTO'},
    'Iran':               {'gdp_ppp': 19000,  'lpi': 2.60, 'inf': 31.0, 'tariff_zone': 'WTO'},
    'Iraq':               {'gdp_ppp': 15000,  'lpi': 2.35, 'inf': 0.4,  'tariff_zone': 'WTO'},
    'Israel':             {'gdp_ppp': 42000,  'lpi': 3.43, 'inf': 0.9,  'tariff_zone': 'WTO'},
    'Ukraine':            {'gdp_ppp': 9000,   'lpi': 2.83, 'inf': 11.0, 'tariff_zone': 'WTO'},
    'Kazakhstan':         {'gdp_ppp': 26000,  'lpi': 2.81, 'inf': 6.0,  'tariff_zone': 'WTO'},
    'Uzbekistan':         {'gdp_ppp': 7000,   'lpi': 2.63, 'inf': 17.0, 'tariff_zone': 'WTO'},
    'Kyrgyzstan':         {'gdp_ppp': 4500,   'lpi': 2.39, 'inf': 1.5,  'tariff_zone': 'WTO'},
    'Georgia':            {'gdp_ppp': 11000,  'lpi': 2.85, 'inf': 2.6,  'tariff_zone': 'WTO'},
    'Mongolia':           {'gdp_ppp': 12000,  'lpi': 2.41, 'inf': 6.8,  'tariff_zone': 'WTO'},
    'Myanmar':            {'gdp_ppp': 6500,   'lpi': 2.30, 'inf': 5.9,  'tariff_zone': 'WTO'},
    'Afghanistan':        {'gdp_ppp': 2000,   'lpi': 1.95, 'inf': 2.3,  'tariff_zone': 'WTO'},
    'Yemen':              {'gdp_ppp': 3500,   'lpi': 2.10, 'inf': 27.0, 'tariff_zone': 'WTO'},
    'Nepal':              {'gdp_ppp': 3000,   'lpi': 2.40, 'inf': 4.2,  'tariff_zone': 'WTO'},

    'India':              {'gdp_ppp': 7000,   'lpi': 3.18, 'inf': 3.9,  'tariff_zone': 'WTO'},
    'Indonesia':          {'gdp_ppp': 12000,  'lpi': 3.15, 'inf': 3.2,  'tariff_zone': 'ASEAN'},
    'Philippines':        {'gdp_ppp': 9000,   'lpi': 3.00, 'inf': 5.2,  'tariff_zone': 'ASEAN'},
    'Thailand':           {'gdp_ppp': 18000,  'lpi': 3.41, 'inf': 1.1,  'tariff_zone': 'ASEAN'},
    'Vietnam':            {'gdp_ppp': 7500,   'lpi': 3.27, 'inf': 3.5,  'tariff_zone': 'ASEAN'},
    'Malaysia':           {'gdp_ppp': 29000,  'lpi': 3.22, 'inf': 1.0,  'tariff_zone': 'ASEAN'},
    'Singapore':          {'gdp_ppp': 105000, 'lpi': 4.00, 'inf': 0.6,  'tariff_zone': 'ASEAN'},
    'Bangladesh':         {'gdp_ppp': 4500,   'lpi': 2.58, 'inf': 5.8,  'tariff_zone': 'WTO'},
    'Pakistan':           {'gdp_ppp': 5500,   'lpi': 2.83, 'inf': 3.9,  'tariff_zone': 'WTO'},

    'Nigeria':            {'gdp_ppp': 5000,   'lpi': 2.50, 'inf': 12.1, 'tariff_zone': 'AFRICA'},
    'South Africa':       {'gdp_ppp': 13000,  'lpi': 3.38, 'inf': 4.6,  'tariff_zone': 'AFRICA'},
    'Egypt':              {'gdp_ppp': 14000,  'lpi': 3.10, 'inf': 10.0, 'tariff_zone': 'AFRICA'},
    'Algeria':            {'gdp_ppp': 13000,  'lpi': 2.80, 'inf': 8.0,  'tariff_zone': 'AFRICA'},
    'Morocco':            {'gdp_ppp': 9500,   'lpi': 2.95, 'inf': 1.2,  'tariff_zone': 'AFRICA'},
    'Sudan':              {'gdp_ppp': 4500,   'lpi': 2.40, 'inf': 50.0, 'tariff_zone': 'AFRICA'},
    'Ivory Coast':        {'gdp_ppp': 6000,   'lpi': 2.50, 'inf': 2.0,  'tariff_zone': 'AFRICA'},
    'Ghana':              {'gdp_ppp': 4500,   'lpi': 2.66, 'inf': 9.8,  'tariff_zone': 'AFRICA'},
    'Senegal':            {'gdp_ppp': 3500,   'lpi': 2.72, 'inf': 0.5,  'tariff_zone': 'AFRICA'},
    'Ethiopia':           {'gdp_ppp': 2200,   'lpi': 2.59, 'inf': 13.0, 'tariff_zone': 'AFRICA'},
    'Tanzania':           {'gdp_ppp': 3000,   'lpi': 2.60, 'inf': 3.5,  'tariff_zone': 'AFRICA'},
    'Kenya':              {'gdp_ppp': 3500,   'lpi': 2.81, 'inf': 4.7,  'tariff_zone': 'AFRICA'},
    'Angola':             {'gdp_ppp': 7000,   'lpi': 2.05, 'inf': 19.6, 'tariff_zone': 'AFRICA'},
    'Mozambique':         {'gdp_ppp': 1300,   'lpi': 2.17, 'inf': 3.9,  'tariff_zone': 'AFRICA'},
    'Zambia':             {'gdp_ppp': 4000,   'lpi': 2.28, 'inf': 7.0,  'tariff_zone': 'AFRICA'},
    'Rwanda':             {'gdp_ppp': 2100,   'lpi': 2.76, 'inf': 0.8,  'tariff_zone': 'AFRICA'},
    'Cameroon':           {'gdp_ppp': 3600,   'lpi': 2.27, 'inf': 0.9,  'tariff_zone': 'AFRICA'},
    'Somalia':            {'gdp_ppp': 500,    'lpi': 1.50, 'inf': 5.0,  'tariff_zone': 'AFRICA'},
    'Liberia':            {'gdp_ppp': 1700,   'lpi': 2.00, 'inf': 23.4, 'tariff_zone': 'AFRICA'},
    'Togo':               {'gdp_ppp': 2600,   'lpi': 2.54, 'inf': 0.9,  'tariff_zone': 'AFRICA'},
    'Benin':              {'gdp_ppp': 2300,   'lpi': 2.59, 'inf': 1.7,  'tariff_zone': 'AFRICA'},
    'Niger':              {'gdp_ppp': 1200,   'lpi': 2.07, 'inf': 6.4,  'tariff_zone': 'AFRICA'},
    'Guinea':             {'gdp_ppp': 2400,   'lpi': 1.98, 'inf': 9.8,  'tariff_zone': 'AFRICA'},
    'Madagascar':         {'gdp_ppp': 1700,   'lpi': 2.18, 'inf': 8.6,  'tariff_zone': 'AFRICA'},
    'Democratic Republic of the Congo': {'gdp_ppp': 800, 'lpi': 1.87, 'inf': 30.0, 'tariff_zone': 'AFRICA'},
}

DEFAULT_STATS_2018 = {'gdp_ppp': 15000, 'lpi': 2.8, 'inf': 3.0, 'tariff_zone': 'WTO'}

COUNTRY_STATS_CURRENT = {
    'USA':                {'gdp_ppp': 85000,  'lpi': 4.00, 'inf': 2.8,  'tariff_zone': 'USMCA'},
    'France':             {'gdp_ppp': 58000,  'lpi': 3.90, 'inf': 2.1,  'tariff_zone': 'EU'},
    'Germany':            {'gdp_ppp': 68000,  'lpi': 4.30, 'inf': 2.0,  'tariff_zone': 'EU'},
    'UK':                 {'gdp_ppp': 57000,  'lpi': 4.05, 'inf': 2.3,  'tariff_zone': 'WTO'},
    'Italy':              {'gdp_ppp': 51000,  'lpi': 3.80, 'inf': 1.5,  'tariff_zone': 'EU'},
    'Spain':              {'gdp_ppp': 50000,  'lpi': 3.90, 'inf': 1.8,  'tariff_zone': 'EU'},
    'Netherlands':        {'gdp_ppp': 73000,  'lpi': 4.15, 'inf': 2.0,  'tariff_zone': 'EU'},
    'Canada':             {'gdp_ppp': 62000,  'lpi': 4.00, 'inf': 2.5,  'tariff_zone': 'USMCA'},
    'Mexico':             {'gdp_ppp': 24000,  'lpi': 3.20, 'inf': 4.5,  'tariff_zone': 'USMCA'},
    'Brazil':             {'gdp_ppp': 18000,  'lpi': 3.10, 'inf': 4.8,  'tariff_zone': 'LATAM'},
    'China':              {'gdp_ppp': 23000,  'lpi': 3.80, 'inf': 2.0,  'tariff_zone': 'WTO'},
    'Japan':              {'gdp_ppp': 52000,  'lpi': 4.05, 'inf': 2.5,  'tariff_zone': 'WTO'},
    'South Korea':        {'gdp_ppp': 50000,  'lpi': 3.95, 'inf': 2.0,  'tariff_zone': 'WTO'},
    'Australia':          {'gdp_ppp': 65000,  'lpi': 3.85, 'inf': 2.5,  'tariff_zone': 'WTO'},
    'India':              {'gdp_ppp': 11000,  'lpi': 3.40, 'inf': 5.0,  'tariff_zone': 'WTO'},
    'Dominican Republic': {'gdp_ppp': 22000,  'lpi': 3.00, 'inf': 4.0,  'tariff_zone': 'LATAM'},
    'Philippines':        {'gdp_ppp': 12000,  'lpi': 3.20, 'inf': 4.5,  'tariff_zone': 'ASEAN'},
    'Turkey':             {'gdp_ppp': 38000,  'lpi': 3.40, 'inf': 20.0, 'tariff_zone': 'WTO'},
    'Russia':             {'gdp_ppp': 30000,  'lpi': 2.90, 'inf': 7.0,  'tariff_zone': 'WTO'},
}

DEFAULT_STATS_CURRENT = {'gdp_ppp': 25000, 'lpi': 3.2, 'inf': 3.5, 'tariff_zone': 'WTO'}

REAL_FUEL_PRICES_2018 = {
    2015: [47.76, 58.10, 55.89, 59.52, 64.08, 61.48, 56.56, 46.52, 47.62, 48.43, 44.27, 38.01],
    2016: [30.70, 32.18, 38.21, 41.58, 46.83, 48.25, 45.07, 45.84, 46.57, 49.52, 44.73, 53.31],
    2017: [54.58, 54.87, 51.59, 52.31, 50.33, 46.37, 48.48, 51.70, 56.15, 57.51, 62.71, 64.37],
    2018: [69.08, 65.32, 66.02, 71.63, 76.98, 75.70, 74.35, 72.58, 78.89, 81.03, 64.75, 57.36]
}
REAL_FUEL_PRICE_CURRENT = 82.0

CATEGORY_TARIFFS = {
    'Electronics': 0.00,
    'Computers':   0.00,
    'Clothing':    0.12,
    'Sporting':    0.08,
    'Toys':        0.05,
    'Furniture':   0.10,
    'Other':       0.05
}

SHIPPING_MODE_COST = {
    'Same Day': 3.0,
    'First Class': 2.0,
    'Second Class': 1.2,
    'Standard Class': 1.0,
}

SHIPPING_MODE_RANK = {
    'Same Day': 0,
    'First Class': 1,
    'Second Class': 2,
    'Standard Class': 3,
}

COUNTRY_COORDS_FALLBACK = {
    'USA':           (38.0,   -97.0),
    'France':        (46.0,     2.0),
    'Mexico':        (23.0,  -102.0),
    'Germany':       (51.0,     9.0),
    'Australia':     (-27.0,  133.0),
    'Brazil':        (-10.0,  -55.0),
    'UK':            (54.0,    -2.0),
    'China':         (35.0,   105.0),
    'Italy':         (42.0,    12.0),
    'India':         (20.0,    77.0),
    'Indonesia':     (-5.0,   120.0),
    'Spain':         (40.0,    -4.0),
    'Turkey':        (39.0,    35.0),
    'Nigeria':       (10.0,     8.0),
    'Puerto Rico':   (18.25,  -66.03),
    'Japan':         (36.0,   138.0),
    'South Korea':   (37.0,   127.0),
    'Singapore':     (1.35,   103.8),
    'Morocco':       (32.0,    -6.0),
    'Sudan':         (15.0,    30.0),
    'Ivory Coast':   (8.0,     -5.0),
    'Egypt':         (27.0,    30.0),
    'Algeria':       (28.0,     3.0),
    'Netherlands':   (52.0,     5.0),
    'Dominican Republic': (19.0, -70.67),
    'Philippines':   (13.0,   122.0),
    'Russia':        (60.0,    90.0),
    'Canada':        (56.0,   -96.0),
    'Argentina':     (-34.0,  -64.0),
    'South Africa':  (-29.0,   25.0),
    'New Zealand':   (-42.0,  172.0),
    'Saudi Arabia':  (24.0,    45.0),
    'Poland':        (52.0,    20.0),
    'Ukraine':       (49.0,    32.0),
    'Thailand':      (15.0,   101.0),
    'Vietnam':       (16.0,   108.0),
    'Malaysia':      (4.0,    109.0),
    'Pakistan':      (30.0,    70.0),
    'Bangladesh':    (24.0,    90.0),
    'Colombia':      (4.0,    -72.0),
    'Chile':         (-30.0,  -71.0),
    'Peru':          (-10.0,  -76.0),
    'Venezuela':     (8.0,    -66.0),
    'Kenya':         (-1.0,    38.0),
    'Ethiopia':      (9.0,    40.0),
    'Tanzania':      (-6.0,   35.0),
    'Ghana':         (8.0,    -1.0),
    'Angola':        (-12.0,  18.0),
}
