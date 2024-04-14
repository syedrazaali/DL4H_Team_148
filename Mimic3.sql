COPY chartevents FROM 'C:\Users\Raza_Ali\Downloads\CHARTEVENTS.csv' DELIMITER ',' CSV HEADER;
COPY labevents FROM 'C:\Users\Raza_Ali\Downloads\LABEVENTS.csv' DELIMITER ',' CSV HEADER;
COPY noteevents FROM 'C:\Users\Raza_Ali\Downloads\NOTEEVENTS.csv' DELIMITER ',' CSV HEADER;

CREATE TABLE chartevents (
    ROW_ID INT,
    SUBJECT_ID INT,
    HADM_ID INT,
    ICUSTAY_ID INT,
    ITEMID INT,
    CHARTTIME TIMESTAMP,
    STORETIME TIMESTAMP,
    CGID INT,
    VALUE TEXT,
    VALUENUM FLOAT,
    VALUEUOM TEXT,
    WARNING BOOLEAN,
    ERROR BOOLEAN,
    RESULTSTATUS TEXT,
    STOPPED TEXT
);

CREATE TABLE labevents (
    ROW_ID INT,
    SUBJECT_ID INT,
    HADM_ID INT,
    ITEMID INT,
    CHARTTIME TIMESTAMP,
    VALUE TEXT,
    VALUENUM FLOAT,
    VALUEUOM TEXT,
    FLAG TEXT
);


CREATE TABLE noteevents (
    ROW_ID INT,
    SUBJECT_ID INT,
    HADM_ID INT,
    CHARTDATE DATE,
    CHARTTIME TIMESTAMP,
    STORETIME TIMESTAMP,
    CATEGORY TEXT,
    DESCRIPTION TEXT,
    CGID INT,
    ISERROR BOOLEAN,
    TEXT TEXT
);

CREATE TABLE d_items (
    ROW_ID INT,
    ITEMID INT,
    LABEL TEXT,
    ABBREVIATION TEXT,
    DBSOURCE TEXT,
    LINKSTO TEXT,
    CATEGORY TEXT,
    UNITNAME TEXT,
    PARAM_TYPE TEXT,
    CONCEPTID INT
);

CREATE TABLE d_labitems (
    ROW_ID INT,
    ITEMID INT,
    LABEL TEXT,
    FLUID TEXT,
    CATEGORY TEXT,
    LOINC_CODE TEXT
);


SELECT ITEMID, LABEL
FROM d_items di
WHERE LABEL LIKE '%glascow%';

SELECT ITEMID, LABEL
FROM d_labitems dl 
WHERE LABEL LIKE ANY (ARRAY[
    '%Capillary refill rate%',
    '%Diastolic blood pressure%',
    '%Fraction inspired oxygen%',
    '%Glascow coma scale eye opening%',
    '%Glascow coma scale motor response%',
    '%Glascow coma scale total%',
    '%Glascow coma scale verbal response%',
    '%Glucose%',
    '%Heart Rate%',
    '%Height%',
    '%Mean blood pressure%',
    '%Oxygen saturation%',
    '%Respiratory rate%',
    '%Systolic blood pressure%',
    '%Temperature%',
    '%Weight%',
    '%pH%'
]);

SELECT ITEMID, LABEL
FROM d_items 
WHERE LABEL LIKE ANY (ARRAY[
    '%Capillary refill rate%',
    '%Diastolic blood pressure%',
    '%Fraction inspired oxygen%',
    '%Glascow coma scale eye opening%',
    '%Glascow coma scale motor response%',
    '%Glascow coma scale total%',
    '%Glascow coma scale verbal response%',
    '%Glucose%',
    '%Heart Rate%',
    '%Height%',
    '%Mean blood pressure%',
    '%Oxygen saturation%',
    '%Respiratory rate%',
    '%Systolic blood pressure%',
    '%Temperature%',
    '%Weight%',
    '%pH%'
]);


SELECT SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE, VALUENUM, VALUEUOM
FROM CHARTEVENTS
WHERE ITEMID IN (1126, 580, 211, 220045, 50809, 50820, 1352, 1394, 645, 676, 677, 678, 679, 733, 763, 780, 807, 811, 3447, 3494, 1880, 3744, 3745, 3777, 3816, 1529, 3580, 3581, 3582, 3583, 3692, 3693, 3723, 8385, 8387, 8537, 45271, 224027, 227054, 226846, 227015, 227016, 228388, 224674, 227586, 225664, 224639, 224642, 227976, 227977, 227978, 227979, 228232, 228242, 220621, 220045, 220047, 227854, 226512, 226531, 226537, 226329, 226707, 226730, 223761, 223762, 50809, 50820, 50825, 50831, 50842, 50931, 51014, 51022, 51034, 51053, 51084, 51094, 51478, 51491, 51529);


SELECT SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE, VALUENUM, VALUEUOM, FLAG
FROM labevents
WHERE ITEMID IN (1126, 580, 211, 220045, 50809, 50820, 1352, 1394, 645, 676, 677, 678, 679, 733, 763, 780, 807, 811, 3447, 3494, 1880, 3744, 3745, 3777, 3816, 1529, 3580, 3581, 3582, 3583, 3692, 3693, 3723, 8385, 8387, 8537, 45271, 224027, 227054, 226846, 227015, 227016, 228388, 224674, 227586, 225664, 224639, 224642, 227976, 227977, 227978, 227979, 228232, 228242, 220621, 220045, 220047, 227854, 226512, 226531, 226537, 226329, 226707, 226730, 223761, 223762, 50809, 50820, 50825, 50831, 50842, 50931, 51014, 51022, 51034, 51053, 51084, 51094, 51478, 51491, 51529);


CREATE TABLE IF NOT EXISTS relevant_ids AS
SELECT DISTINCT ce.SUBJECT_ID, ce.HADM_ID
FROM chartevents ce
INNER JOIN labevents le ON ce.SUBJECT_ID = le.SUBJECT_ID AND ce.HADM_ID = le.HADM_ID;

CREATE INDEX idx_noteevents ON public.noteevents USING btree (subject_id, hadm_id);
CREATE INDEX idx_admissions_subject_id_hadm_id ON public.admissions (subject_id, hadm_id);
CREATE INDEX idx_patients_subject_id ON public.patients (subject_id);
CREATE INDEX idx_icustays_subject_id_hadm_id ON public.icustays (subject_id, hadm_id);
CREATE INDEX idx_chartevents_subject_id_hadm_id ON public.chartevents (subject_id, hadm_id);
CREATE INDEX idx_labevents_subject_id_hadm_id ON public.labevents (subject_id, hadm_id);


CREATE TABLE merged_data AS
SELECT
    a.SUBJECT_ID,
    a.HADM_ID,
    a.ADMITTIME,
    a.DISCHTIME,
    a.DEATHTIME,
    p.GENDER,
    EXTRACT(YEAR FROM age(a.ADMITTIME, p.DOB)) AS age_at_admission,
    a.DIAGNOSIS,
    i.ICUSTAY_ID,
    i.INTIME AS icu_intime,
    i.OUTTIME AS icu_outtime,
    i.LOS AS icu_los,
    (CASE WHEN a.DEATHTIME IS NOT NULL THEN 1 ELSE 0 END) AS mortality_label
FROM
    admissions a
JOIN
    relevant_ids ri ON a.SUBJECT_ID = ri.SUBJECT_ID AND a.HADM_ID = ri.HADM_ID
JOIN
    patients p ON a.SUBJECT_ID = p.SUBJECT_ID
LEFT JOIN
    icustays i ON a.SUBJECT_ID = i.SUBJECT_ID AND a.HADM_ID = i.HADM_ID;
    
   
-- including note events in the merged_data:
CREATE TABLE merged_data AS
SELECT
    a.SUBJECT_ID,
    a.HADM_ID,
    a.ADMITTIME,
    a.DISCHTIME,
    a.DEATHTIME,
    p.GENDER,
    EXTRACT(YEAR FROM age(a.ADMITTIME, p.DOB)) AS age_at_admission,
    a.DIAGNOSIS,
    i.ICUSTAY_ID,
    i.INTIME AS icu_intime,
    i.OUTTIME AS icu_outtime,
    i.LOS AS icu_los,
    (CASE WHEN a.DEATHTIME IS NOT NULL THEN 1 ELSE 0 END) AS mortality_label,
    MAX(ce.valuenum) AS max_chart_value,  -- Example aggregation from chartevents
    AVG(le.valuenum) AS avg_lab_value,    -- Example aggregation from labevents
    STRING_AGG(ne.text, ' ') AS all_notes -- Aggregate all notes into a single string
FROM
    admissions a
JOIN
    patients p ON a.SUBJECT_ID = p.SUBJECT_ID
LEFT JOIN
    icustays i ON a.SUBJECT_ID = i.SUBJECT_ID AND a.HADM_ID = i.HADM_ID
LEFT JOIN
    chartevents ce ON a.SUBJECT_ID = ce.SUBJECT_ID AND a.HADM_ID = ce.HADM_ID
LEFT JOIN
    labevents le ON a.SUBJECT_ID = le.SUBJECT_ID AND a.HADM_ID = le.HADM_ID
LEFT JOIN
    noteevents ne ON a.SUBJECT_ID = ne.SUBJECT_ID AND a.HADM_ID = ne.HADM_ID
GROUP BY
    a.SUBJECT_ID, a.HADM_ID, a.ADMITTIME, a.DISCHTIME, a.DEATHTIME, p.GENDER, p.DOB,
    a.DIAGNOSIS, i.ICUSTAY_ID, i.INTIME, i.OUTTIME, i.LOS;


   -- temp tables: 
CREATE TEMP TABLE pre_agg_chartevents AS
SELECT
    subject_id,
    hadm_id,
    MAX(valuenum) AS max_chart_value
FROM
    chartevents
GROUP BY
    subject_id, hadm_id;
   
   
CREATE TEMP TABLE pre_agg_labevents AS
SELECT
    subject_id,
    hadm_id,
    AVG(valuenum) AS avg_lab_value
FROM
    labevents
GROUP BY
    subject_id, hadm_id;

   
CREATE TEMP TABLE pre_agg_noteevents AS
SELECT
    subject_id,
    hadm_id,
    STRING_AGG(text, ' ') AS all_notes
FROM
    noteevents
GROUP BY
    subject_id, hadm_id;

-- trying to build the merged_data table after breaking the query into parts 
CREATE TABLE merged_data AS
SELECT
    a.SUBJECT_ID,
    a.HADM_ID,
    a.ADMITTIME,
    a.DISCHTIME,
    a.DEATHTIME,
    p.GENDER,
    EXTRACT(YEAR FROM age(a.ADMITTIME, p.DOB)) AS age_at_admission,
    a.DIAGNOSIS,
    i.ICUSTAY_ID,
    i.INTIME AS icu_intime,
    i.OUTTIME AS icu_outtime,
    i.LOS AS icu_los,
    (CASE WHEN a.DEATHTIME IS NOT NULL THEN 1 ELSE 0 END) AS mortality_label,
    ce.max_chart_value,
    le.avg_lab_value,
    ne.all_notes
FROM
    admissions a
JOIN
    patients p ON a.SUBJECT_ID = p.SUBJECT_ID
LEFT JOIN
    icustays i ON a.SUBJECT_ID = i.SUBJECT_ID AND a.HADM_ID = i.HADM_ID
LEFT JOIN
    pre_agg_chartevents ce ON a.SUBJECT_ID = ce.SUBJECT_ID AND a.HADM_ID = ce.HADM_ID
LEFT JOIN
    pre_agg_labevents le ON a.SUBJECT_ID = le.SUBJECT_ID AND a.HADM_ID = le.HADM_ID
LEFT JOIN
    pre_agg_noteevents ne ON a.SUBJECT_ID = ne.SUBJECT_ID AND a.HADM_ID = ne.HADM_ID;

 