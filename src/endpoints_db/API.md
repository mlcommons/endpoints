# MLPerf Submission API



# Base URL


| URL | Description |
|-----|-------------|


# APIs

## POST /runs

Create Run



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| user_id | string | True | PRISM user ID of the submitter |


### Request Body

[RunCreate](#runcreate)







### Responses

#### 201


Successful Response


[RunOut](#runout)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /runs

List Runs



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| user_id | string | True | PRISM user ID |


### Responses

#### 200


Successful Response


array







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /runs/{run_id}

Get Run



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| run_id | string | True |  |
| X-User-Id | string | True |  |


### Responses

#### 200


Successful Response


[RunOut](#runout)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /runs/{run_id}

Delete Run



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| run_id | string | True |  |
| X-User-Id | string | True |  |


### Responses

#### 200


Successful Response








#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PATCH /runs/{run_id}/pin

Pin Run



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| run_id | string | True |  |
| X-User-Id | string | True |  |


### Responses

#### 200


Successful Response


[RunOut](#runout)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PATCH /runs/{run_id}/unpin

Unpin Run



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| run_id | string | True |  |
| X-User-Id | string | True |  |


### Responses

#### 200


Successful Response


[RunOut](#runout)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /submissions

Create Submission



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| user_id | string | True | PRISM user ID of the submitter |


### Request Body

[SubmissionCreate](#submissioncreate)







### Responses

#### 201


Successful Response


[SubmissionOut](#submissionout)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /submissions

List Submissions



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| user_id | string | True | PRISM user ID |


### Responses

#### 200


Successful Response


array







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /submissions/{submission_id}

Get Submission



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| submission_id | string | True |  |
| user_id | string | True | PRISM user ID |
| include_runs | boolean | False |  |


### Responses

#### 200


Successful Response


[SubmissionWithRuns](#submissionwithruns)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## PATCH /submissions/{submission_id}

Update Submission



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| submission_id | string | True |  |
| user_id | string | True | PRISM user ID |


### Request Body

[SubmissionUpdate](#submissionupdate)







### Responses

#### 200


Successful Response


[SubmissionOut](#submissionout)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /submissions/{submission_id}

Withdraw Submission



### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| submission_id | string | True |  |
| user_id | string | True | PRISM user ID |


### Responses

#### 200


Successful Response


[SubmissionOut](#submissionout)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /health

Health





### Responses

#### 200


Successful Response








# Components



## HTTPValidationError



| Field | Type | Description |
|-------|------|-------------|
| detail | array |  |


## RunCreate



| Field | Type | Description |
|-------|------|-------------|
| started_at | string |  |
| finished_at | string |  |
| expires_at |  |  |
| pinned | boolean |  |
| system_info | object |  |
| config | object |  |
| result_summary | object |  |
| archive_uri | string |  |


## RunOut



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| user_id | string |  |
| started_at | string |  |
| finished_at | string |  |
| expires_at |  |  |
| pinned | boolean |  |
| system_info | object |  |
| config | object |  |
| result_summary | object |  |
| archive_uri | string |  |


## RunSummary



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| model |  |  |
| concurrency |  |  |
| started_at | string |  |
| finished_at | string |  |


## SubmissionCreate



| Field | Type | Description |
|-------|------|-------------|
| benchmark_version | integer |  |
| division | string |  |
| availability | string |  |
| early_publish | boolean |  |
| publication_cycle |  |  |
| target_availability_date |  |  |
| run_ids | array |  |


## SubmissionOut



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| user_id | string |  |
| created_at | string |  |
| status | string |  |
| benchmark_version | integer |  |
| division | string |  |
| availability | string |  |
| early_publish | boolean |  |
| publication_cycle |  |  |
| target_availability_date |  |  |
| availability_qualified_at |  |  |
| compliance_passed_at |  |  |
| first_published_at |  |  |
| peer_review_started_at |  |  |
| objection_resolution_started_at |  |  |
| finalized_at |  |  |
| withdrawn_at |  |  |
| run_ids | array |  |
| archive_uri |  |  |
| pr_url |  |  |
| pr_number |  |  |


## SubmissionUpdate



| Field | Type | Description |
|-------|------|-------------|
| status |  |  |
| availability_qualified_at |  |  |
| compliance_passed_at |  |  |
| first_published_at |  |  |
| peer_review_started_at |  |  |
| objection_resolution_started_at |  |  |
| finalized_at |  |  |
| pr_url |  |  |
| pr_number |  |  |
| archive_uri |  |  |
| publication_cycle |  |  |
| target_availability_date |  |  |


## SubmissionWithRuns



| Field | Type | Description |
|-------|------|-------------|
| id | string |  |
| user_id | string |  |
| created_at | string |  |
| status | string |  |
| benchmark_version | integer |  |
| division | string |  |
| availability | string |  |
| early_publish | boolean |  |
| publication_cycle |  |  |
| target_availability_date |  |  |
| availability_qualified_at |  |  |
| compliance_passed_at |  |  |
| first_published_at |  |  |
| peer_review_started_at |  |  |
| objection_resolution_started_at |  |  |
| finalized_at |  |  |
| withdrawn_at |  |  |
| run_ids | array |  |
| archive_uri |  |  |
| pr_url |  |  |
| pr_number |  |  |
| runs | array |  |


## ValidationError



| Field | Type | Description |
|-------|------|-------------|
| loc | array |  |
| msg | string |  |
| type | string |  |
