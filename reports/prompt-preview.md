# Rendered prompt examples — for W1–W3 review (SPEC §12)

## Official generation prompt (B4 / candidates; single official wording)


```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
marital_status, child_number, age_first_childbirth, education, laborforce, occupation, spouse_occupation, income, wealth, health, depress, isolated, lonely, mental_health, gender_role_attitude, happy, political_view, satisfy_job, trust, vocabulary_test, work_hard.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- marital_status (static): Marital status
- child_number (static): Number of Child ever born
- age_first_childbirth (static): Age at first childbirth
- education (static): Educational attainment
- laborforce (static): Whether the respondent participated in the labor force
- occupation (static): Occupation
- spouse_occupation (static): Spouse's occupation
- income (static): Income from occupation last year
- wealth (static): Total personal wealth
- health (static): Self-evaluated health condition
- depress (static): Ever had depression or not
- isolated (static): how often in the past 4 weeks has felt isolated from others
- lonely (static): how often in the past 4 weeks has felt that lack companionship
- mental_health (static): how many days during the past 30 days was mental health condition not good
- gender_role_attitude (static): Attitude toward gender roles, indicated by agreeing or disagreeing the sentence "It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family."
- happy (static): General happiness
- political_view (static): Political view
- satisfy_job (static): Satisfaction with work
- trust (static): Trust with people
- vocabulary_test (static): Total number of correct words in a vocabulary test, which is an indicator of the level of cognitive skills
- work_hard (static): Evaluation on the importance of working hard, which is an indicator of the level of noncognitive skills

Allowed values:
- marital_status: ["Married", "Divorced/widowed", "Single"]
- child_number: number in [0, 8]
- age_first_childbirth: number in [17, 45] OR one of ['No Child']
- education: ["Less than high school", "High school", "College and above"]
- laborforce: ["Not in the labor force", "In the labor force"]
- occupation: ["Management, professional, and related occupations", "Service occupations", "Sales and office occupations", "Natural resources, construction, and maintenance occupations", "Production, transportation, and material moving occupations", "Military Occupations", "Unemployed"]
- spouse_occupation: ["Management, professional, and related occupations", "Service occupations", "Sales and office occupations", "Natural resources, construction, and maintenance occupations", "Production, transportation, and material moving occupations", "Military Occupations", "No spouse"]
- income: ["LT $5000", "$5000 TO 9999", "$10000 OR MORE", "Unemployed"]
- wealth: ["LT $40000", "$40000 TO 500000", "More than $500000"]
- health: ["Excellent", "Good", "Fair", "Poor"]
- depress: ["Had depression", "No depression"]
- isolated: ["Never", "Rarely", "Sometimes", "Often", "Very often"]
- lonely: ["Never", "Rarely", "Sometimes", "Often", "Very often"]
- mental_health: number in [0, 30]
- gender_role_attitude: ["Agree", "Disagree"]
- happy: ["Very happy", "Pretty happy", "Not too happy"]
- political_view: ["Liberal", "Middle", "Conservative"]
- satisfy_job: ["Very satisfied", "Moderately satisfied", "A little dissatisfied", "Very dissatisfied"]
- trust: ["Most people can be trusted", "Can't be too careful", "Depends"]
- vocabulary_test: number in [0, 10]
- work_hard: ["MOST IMPORTANT", "2ND IMPORTANT", "3RD IMPORTANT", "4TH IMPORTANT", "LEAST IMPORTANT"]
```

## Family 1 — judge candidate values (call J)

### W1
```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
political_view.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- political_view (static): Political view

Allowed values:
- political_view: ["Liberal", "Middle", "Conservative"]

Below are 3 possible answers to the question "Political view" for a person with exactly the attributes above. For each answer, give the probability that a real, randomly selected respondent with these attributes in this population gave that answer. The probabilities over the listed answers must sum to 1. Do not favour an answer because it seems typical; favour it only if it is actually more common among such people.

Candidate answers:
A: Liberal
B: Conservative
C: Middle

Return JSON only, one key per label, e.g. {"A": 0.4, "B": 0.6}.
```

### W2
```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
political_view.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- political_view (static): Political view

Allowed values:
- political_view: ["Liberal", "Middle", "Conservative"]

Imagine 100 real respondents from this population who all have exactly the attributes above. Below are 3 possible answers to the question "Political view". For each answer, say how many of the 100 would have given it. The numbers must sum to 100. Answers that only some of these people would give should get the share those people represent, not zero.

Candidate answers:
A: Liberal
B: Conservative
C: Middle

Return JSON only, one key per label with a count out of 100, e.g. {"A": 40, "B": 60}.
```

### W3
```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
political_view.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- political_view (static): Political view

Allowed values:
- political_view: ["Liberal", "Middle", "Conservative"]

You are a survey researcher who has studied this population closely. For a respondent with exactly the attributes above, the question "Political view" has 3 candidate answers listed below. Based on what you know about how such people actually answered, estimate for each candidate the probability that it is this respondent's real answer. The probabilities must sum to 1.

Candidate answers:
A: Liberal
B: Conservative
C: Middle

Return JSON only, one key per label, e.g. {"A": 0.4, "B": 0.6}.
```

## Family 2 — judge whole profiles (P1/P2/P3/P5; 21-output profiles, abbreviated here)

### W1
```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
marital_status, child_number.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- marital_status (static): Marital status
- child_number (static): Number of Child ever born

Allowed values:
- marital_status: ["Married", "Divorced/widowed", "Single"]
- child_number: number in [0, 8]

For people with exactly these input attributes in this population, how likely is each profile below to be the answers of a real, randomly selected respondent? Give a probability for each profile; the probabilities must sum to 1. Judge each profile as a whole: whether its answers fit the inputs and fit each other.

Candidate profiles:
A: {"marital_status": "Married", "child_number": "2"}
B: {"marital_status": "Married", "child_number": "2"}

Return JSON only, one key per label, e.g. {"A": 0.4, "B": 0.6}.
```

## Family 3 — plausibility filter (S3)

```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
political_view.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- political_view (static): Political view

Allowed values:
- political_view: ["Liberal", "Middle", "Conservative"]

For a person with exactly the attributes above, mark each listed answer as "plausible" if a real respondent with these attributes could reasonably have given it, or "implausible" if it would be very unusual or contradictory. Mark an answer implausible only when you are confident; uncommon is not the same as implausible.

Candidate answers:
A: Middle
B: Conservative
C: Liberal

Return JSON only, one key per label with the value "plausible" or "implausible".
```

## Family 4 — pairwise (S5)

```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
political_view.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- political_view (static): Political view

Allowed values:
- political_view: ["Liberal", "Middle", "Conservative"]

For a person with exactly the attributes above, which of the two answers below is more likely to be a real respondent's answer to "Political view"? Give the probability that X is the real one rather than Y.

X: Liberal
Y: Conservative

Return JSON only: {"X": p} (W2: a count out of 100).
```

## Family 5 — stated distribution for one person (S6/B5/R3; mental_health shows numeric keys)

### W1 (mental_health)
```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
mental_health.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- mental_health (static): how many days during the past 30 days was mental health condition not good

Allowed values:
- mental_health: number in [0, 30]

For a person with exactly the attributes above, give the probability of each allowed answer to "how many days during the past 30 days was mental health condition not good". The probabilities must sum to 1 and must cover every allowed answer, including those that are uncommon. The distribution should reflect how real respondents with these attributes actually answered, not the single most typical answer.

Allowed answers (use exactly these as the JSON keys): ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]

Return JSON only, one key per allowed answer, exactly as listed.
```

### W2 (mental_health)
```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
mental_health.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- mental_health (static): how many days during the past 30 days was mental health condition not good

Allowed values:
- mental_health: number in [0, 30]

Imagine 100 real respondents from this population with exactly the attributes above. For "how many days during the past 30 days was mental health condition not good", say how many of the 100 gave each allowed answer. Every allowed answer must have a number (0 is allowed); the numbers must sum to 100.

Allowed answers (use exactly these as the JSON keys): ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]

Return JSON only, one key per allowed answer, exactly as listed.
```

### W3 (mental_health)
```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Conditioned on the following input attributes:
birth_year: 1975, age: 43, gender: Male, race: Non-Black, Non-Hispanic, immigrant_status: Non-immigrant, mother_education: High school, father_education: High school, mother_occupation: Sales and office occupations, father_occupation: Sales and office occupations
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
mental_health.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- mental_health (static): how many days during the past 30 days was mental health condition not good

Allowed values:
- mental_health: number in [0, 30]

You are a survey researcher who has studied this population. For respondents with exactly the attributes above, estimate the distribution of answers to "how many days during the past 30 days was mental health condition not good" as you would expect to find it in the actual survey data: a probability for every allowed answer, summing to 1.

Allowed answers (use exactly these as the JSON keys): ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]

Return JSON only, one key per allowed answer, exactly as listed.
```

## Family 6 — stated two-way table row (S7/P4)

### W1
```
You are simulating one *randomly selected* synthetic individual sampled from Adult population (18+) in the United States in 2018.
The data represent a cross-sectional snapshot collected in the year 2018.
Return output strictly as JSON. Do not include explanations, markdown fences, or additional keys.
The JSON object must contain EXACTLY these keys:
political_view.

Rules:
1) Use ONLY the allowed options below for each field.
2) For numeric fields, use numbers within the specified range or one of the special labels.


Descriptions:
- political_view (static): Political view

Allowed values:
- political_view: ["Liberal", "Middle", "Conservative"]

Among women aged 35-54 in this population in the reference year, what share gave each allowed answer to "Political view"? Give a probability for every allowed answer, summing to 1. Base this on the actual distribution among such people, including minority answers.

Allowed answers (use exactly these as the JSON keys): ["Liberal", "Middle", "Conservative"]

Return JSON only, one key per allowed answer, exactly as listed.
```
