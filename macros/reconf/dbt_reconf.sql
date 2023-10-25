/*
dbt reconfigured macros before this is available as dbt package
*/
{%- macro q(col) -%}
{{ adapter.quote(col) }}
{%- endmacro -%}
{%- macro t(type_) -%}
{{ api.Column.translate_type(type_) }}
{%- endmacro -%}

--
{% macro case_when() %}
{{ return(adapter.dispatch('case_when')(varargs)) }}
{% endmacro %}

{% macro default__case_when(cases) %}
CASE
{% for case in cases %}{{ case }}{% endfor %}
END
{% endmacro %}

{% macro if_then(when, then) %}
{{ return(adapter.dispatch('if_then')(when, then)) }}
{%- endmacro %}

{% macro default__if_then(when, then) %}
  WHEN {{ when }}
  THEN {{ then }}
{% endmacro %}

{% macro else_(then) %}
{{ return(adapter.dispatch('else_')(then)) }}
{%- endmacro %}

{% macro default__else_(then) %}
  ELSE {{ then }}
{% endmacro %}
--

{%- macro star_ref(table_alias) -%}
{{ return(adapter.dispatch('star_ref')(table_alias)) }}
{% endmacro %}
{%- macro default__star_ref(table_alias) -%}
exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{%- macro bigquery__star_ref(table_alias) -%}
{{ table_alias }}
{% endmacro %}
{%- macro snowflake__star_ref(table_alias) -%}
{{ table_alias }}.*
{% endmacro %}
{%- macro redshift__star_ref(table_alias) -%}
{{ table_alias }}.*
{% endmacro %}

{%- macro if_incr(stmt, fallback) -%}
{%- if is_incremental() -%}{{ stmt }}{%-else %}{{ fallback }}{% endif -%}
{% endmacro %}

{%- macro min_ts() -%}
CAST('0001-01-01T00:00:00' AS TIMESTAMP)
{%- endmacro -%}

{%- macro run_start_ts() -%}
CAST({{ string_literal(run_started_at.isoformat()) }} AS {{ t('timestamp') }})
{%- endmacro -%}

{%- macro hash_from_cols(columns, alias=False) -%}
{{ return(adapter.dispatch('hash_from_cols')(columns, alias)) }}
{% endmacro %}
{%- macro default__hash_from_cols(columns, alias) -%}
exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}

{%- macro bigquery__hash_from_cols(columns, alias) -%}
TO_HEX(SHA256(TO_JSON_STRING(JSON_OBJECT(
{%- for column in columns -%}
  '{{ column }}',{%- if alias %} {{ alias }}.{%- else %} {% endif %}{{ column }}
  {%- if not loop.last -%}, {% endif -%}
{%- endfor -%}
))))
{% endmacro %}

{%- macro snowflake__hash_from_cols(columns, alias) -%}
SHA2(TO_JSON(OBJECT_CONSTRUCT(
{%- for column in columns -%}
  '{{ column }}',{%- if alias %} {{ alias }}.{%- else %} {% endif %}{{ column }}
  {%- if not loop.last -%}, {% endif -%}
{%- endfor -%}
)), 256)
{% endmacro %}

{%- macro redshift__hash_from_cols(columns, alias) -%}
SHA2(
{%- for column in columns -%}
  '{{ column }}' || '|' || CAST({%- if alias %}{{ alias }}.{% endif %}{{ column }} AS TEXT)
  {%- if not loop.last -%} || '||' || {% endif -%}
{%- endfor -%}
, 256)
{% endmacro %}




/* RCMACRO
normalize_email
 */
{% macro normalize_email(text) %}
  {{ return(adapter.dispatch('normalize_email') (text)) }}
{% endmacro %}
{% macro default__normalize_email() %}
  exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{% macro snowflake__normalize_email(text) %}
  CASE
    WHEN REGEXP_LIKE(
      TRIM({{ text }}),
      '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9\\-.]+$'
    ) THEN REGEXP_REPLACE(LOWER(TRIM({{ text }})), '\\+[\\d\\D]*\\@', '@')
    ELSE NULL
  END
{% endmacro%}
{% macro bigquery__normalize_email(text) %}
  CASE
    WHEN REGEXP_CONTAINS(
      TRIM({{ text }}),
      r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    ) THEN REGEXP_REPLACE(LOWER(TRIM({{ text }})), r'\+[\d\D]*\@', '@')
    ELSE NULL
  END
{% endmacro %}
/* END RCMACRO */
/* RCMACRO
integer_category_column
*/
{% macro integer_category_column(col, categories) %}
  CASE
{% for c in categories -%}
  WHEN {{ col }} = {{ string_literal(c) }} THEN {{ loop.index }}
{% endfor -%}
  END
{% endmacro %}
/* RCMACRO
aggregate
*/
{% macro aggregate(agg, col, order_by, order_by_direction) -%}
  {{ return(adapter.dispatch('aggregate')(agg, col, order_by, order_by_direction)) }}
{% endmacro %}
{% macro default__aggregate(agg, col, order_by, order_by_direction) -%}
{{ agg }}({{ col }}{% if order_by %} order by {{ order_by }}{% if order_by_direction %} {{ order_by_direction }}{%-endif -%}{%- endif -%})
{% endmacro %}
/* RCMACRO
most_recent
*/
{% macro most_recent(col, recency_col) -%}
{{ return(adapter.dispatch('most_recent')(col, recency_col)) }}
{%- endmacro %}
{% macro default__most_recent(col, recency_col) -%}
{{ array_get(array_agg(col, True, False, recency_col, 'DESC'), 0) }}
{%- endmacro %}
{% macro snowflake__most_recent(col, recency_col) -%}
GET(ARRAY_AGG({{ col }}) WITHIN GROUP (ORDER BY {{ recency_col }} DESC), 0)
{%- endmacro %}
{% macro redshift__most_recent(col, recency_col) -%}
SPLIT_PART(LISTAGG({{ col }}::varchar, ', ') WITHIN GROUP (ORDER BY {{ recency_col }} DESC), ', ', 1)
{%- endmacro %}
/* RCMACRO
most_oldest
*/
{% macro most_oldest(col, recency_col) -%}
{{ return(adapter.dispatch('most_oldest')(col, recency_col)) }}
{%- endmacro %}
{% macro default__most_oldest(col, recency_col) -%}
{{ array_get(array_agg(col, True, False, recency_col, 'ASC'), 0) }}
{%- endmacro %}
{% macro redshift__most_oldest(col, recency_col) -%}
SPLIT_PART(LISTAGG({{ col }}::varchar, ', ') WITHIN GROUP (ORDER BY {{ recency_col }} ASC), ', ', 1)
{%- endmacro %}
/* END RCMACRO */
/* RCMACRO
array_agg
*/
{% macro array_agg(col, ignore_nulls, is_distinct, order_by, order_by_direction) -%}
array_agg({% if is_distinct %}distinct {% endif %}{{ col }}{% if ignore_nulls %} ignore nulls{% endif %}
{% if order_by %} order by {{ order_by }}{% if order_by_direction %} {{ order_by_direction }}{%-endif -%}{%- endif -%})
{%- endmacro %}
/* END RCMACRO */
/* RCMACRO
array_get
*/
{% macro array_get(val, i, t) -%}
  {{ return(adapter.dispatch('array_get')(val, i, t)) }}
{% endmacro %}
{% macro default__array_get(val, i, t) -%}
  exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{% macro snowflake__array_get(val, i, t) -%}
{%- if t %}cast({% endif -%}
get({{ val }}, {{ i }})
{%- if t %} as {{ t }}){% endif -%}
{% endmacro %}
{% macro bigquery__array_get(val, i, t) -%}
{%- if t %}cast({% endif -%}
{{ val }}[safe_offset({{ i }})]
{%- if t %} as {{ t }}){% endif -%}
{% endmacro %}
/* END RCMACRO */
/* RCMACRO
to_json
*/
{% macro to_json(d) %}
  {{ return(adapter.dispatch('to_json') (d)) }}
{% endmacro %}
{% macro default__to_json() %}
  exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{% macro snowflake__to_json(d) %}
TO_JSON({{ d }})
{% endmacro%}
{% macro bigquery__to_json(d) %}
TO_JSON({{ d }})
{% endmacro %}
/* END RCMACRO */
/* RCMACRO
to_json
*/
{% macro to_json_string(d) %}
  {{ return(adapter.dispatch('to_json_string') (d)) }}
{% endmacro %}
{% macro default__to_json_string() %}
  exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{% macro bigquery__to_json_string(d) %}
TO_JSON_STRING({{ d }})
{% endmacro %}
{% macro snowflake__to_json_string(d) %}
TO_JSON({{ d }})
{% endmacro%}
/* END RCMACRO */
/* RCMACRO
json_value
*/
{% macro json_value(val, json_path) %}
  {{ return(adapter.dispatch('json_value')(val, json_path)) }}
{% endmacro %}
{% macro default__json_value() %}
  exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{% macro bigquery__json_value(d, json_path) %}
JSON_VALUE({{ d }}, '{{ json_path }}')
{% endmacro %}
{% macro snowflake__json_value(d, json_path) %}
GET_PATH({{ d }}, '{{ json_path | replace('$.', '') }}')
{% endmacro %}
{% macro postgres__json_value(d, json_path) %}
{% set jpath = json_path.replace('$.', '').replace('[', '.').replace(']', '').split('.') %}
{{ d }}
{%- for p in jpath -%}
    {% if loop.last %}->>{% else %}->{% endif -%}
    {% if p.isnumeric() %}{{ p }}{% else %}'{{ p }}'{% endif %}
{%- endfor %}
{% endmacro %}
{% macro redshift__json_value(d, json_path) %}
{% set jpath = json_path.replace('$', '') %}
{{ d }}{{ jpath }}
{% endmacro %}
/* END RCMACRO */
/* RCMACRO
md5
*/
{% macro md5(d) %}
  {{ return(adapter.dispatch('md5') (d)) }}
{% endmacro %}
{% macro default__md5() %}
  exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{% macro redshift__md5(d) %}
MD5({{ d }})
{% endmacro%}
{% macro snowflake__md5(d) %}
MD5({{ d }})
{% endmacro%}
{% macro bigquery__md5(d) %}
TO_HEX(MD5({{ d }}))
{% endmacro %}
/* END RCMACRO */
/* RCMACRO
count_distinct
*/
{% macro count_distinct(d) %}
  {{ return(adapter.dispatch('count_distinct') (d)) }}
{% endmacro %}
{% macro default__count_distinct(d) %}
COUNT(DISTINCT {{ d }})
{% endmacro %}
//* RCMACRO
distinct
*/
{% macro distinct(d) %}
  {{ return(adapter.dispatch('distinct') (d)) }}
{% endmacro %}
{% macro default__distinct(d) %}
DISTINCT {{ d }}
{% endmacro %}
/* END RCMACRO */* END RCMACRO */
/* RCMACRO
in
*/
{% macro in(v, arr) %}
  {{ return(adapter.dispatch('in')(v, arr)) }}
{% endmacro %}
{% macro default__in() %}
  exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{% macro snowflake__in(v, arr) %}
ARRAY_CONTAINS({{ v }}::variant, {{ arr }})
{% endmacro%}
{% macro bigquery__in(v, arr) %}
{{ v }} IN UNNEST({{ arr }})
{% endmacro %}
/* END RCMACRO */
/* RCMACRO
epoch_to_timestamp
*/
{% macro epoch_to_timestamp(d) %}
{{ return(adapter.dispatch('epoch_to_timestamp')(d)) }}
{% endmacro %}
{% macro default__epoch_to_timestamp(d) %}
exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}
{% macro redshift__epoch_to_timestamp(d) %}
TIMESTAMP 'epoch' + {{ d }} * INTERVAL '1 second'
{% endmacro %}
/* END RCMACRO */
/* RCMACRO
parse_website_domain
*/
{% macro parse_website_domain(text) %}
{{ return(adapter.dispatch('parse_website_domain') (text)) }}
{% endmacro %}

{% macro default__parse_website_domain() %}
exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}

{% macro snowflake__parse_website_domain(text) %}
TRIM(LOWER(REGEXP_SUBSTR(
  CAST({{ text }} AS STRING),
  '((http(s)?):\\/\\/)?(www\\.)?([a-zA-Z0-9@:%._\\-\\+~#=]{2,256}\\.[a-z]{2,6})\\b([-a-zA-Z0-9@:%_\\+.~#?&//=]*)',
  1, 1, 'i', 5
)))
{% endmacro %}
{% macro bigquery__parse_website_domain(text) %}
TRIM(LOWER(REGEXP_EXTRACT(
  CAST({{ text }} AS STRING),
  r'(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9@:%._\-\+~#=]{2,256}\.[a-z]{2,6})\b[-a-zA-Z0-9@:%_\+.~#?&//=]*'
)))
{% endmacro %}

{% macro redshift__parse_website_domain(text) %}
TRIM(LOWER(REGEXP_REPLACE(
  {{ text }}::TEXT,
  '((http(s)?):\\/\\/)?(www\\.)?([a-zA-Z0-9@:%._\\-\\+~#=]{2,256}\\.[a-z]{2,6})\\b([-a-zA-Z0-9@:%_\\+.~#?&//=]*)',
  '$5'
)))
{% endmacro %}
/* END RCMACRO */
/* RCMACRO
every_unique
*/
{% macro every_unique(input) %}
  {{ return(adapter.dispatch('every_unique') (input)) }}
{% endmacro %}

{% macro default__every_unique(input) %}
  exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}

{% macro bigquery__every_unique(input) %}
array_agg(distinct {{ input }} ignore nulls)
{% endmacro %}

{% macro snowflake__every_unique(input) %}
array_compact(array_agg(distinct {{ input }}))
{% endmacro%}
/* END RCMACRO */

{% macro parse_email_domain(text) %}
{{ return(adapter.dispatch('parse_email_domain') (text)) }}
{% endmacro %}

{% macro default__parse_email_domain() %}
exceptions.raise_compiler_error("Unsupported target database")
{% endmacro %}

{% macro snowflake__parse_email_domain(text) %}
TRIM(LOWER(REGEXP_SUBSTR(
  CAST({{ text }} AS STRING),
  '.*@([-a-zA-Z0-9\.]*)',
  1, 1, 'i', 1
)))
{% endmacro %}
{% macro bigquery__parse_email_domain(text) %}
TRIM(LOWER(REGEXP_EXTRACT(
  CAST({{ text }} AS STRING),
  r'.*@([-a-zA-Z0-9\.]*)'
)))
{% endmacro %}

{% macro redshift__parse_email_domain(text) %}
TRIM(LOWER(REGEXP_REPLACE(
  {{ text }}::TEXT,
  '.*@([-a-zA-Z0-9\.]*)',
  '$1'
)))
{% endmacro %}
