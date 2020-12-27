Flask and dialogflow -> chatbot python
=== Important
https://www.youtube.com/watch?v=ubub9Nz681s&list=PLrYq9TGMS8e8E4ZTjFKiwsy_EntyNbDEm
https://github.com/AndroidArena/BestCovid19_bot-DialogFlow

https://cloud.google.com/dialogflow/docs/basics
---


save user-bot conversation on database
dialogflow-> will match text with intent -> logic
query -> matched with intent -> then provide answers

steps:
1. google dialogflow & NLU
2. flask/python, api, mongodb..
3. testing & deployment

### google dialogflow
https://dialogflow.cloud.google.com/

1. create agent
2. create intents, entity, training, pharses, action, response
- Default welcome intent
- Default Fallback intent -> whenever bot doesn't understand

- welcome intent ->
- training phrases
- need to create multiple intents

- Responses : text / quick replies (boutons)

- recognize pattern -> via entities -> action

3. create knowledge base
- as csv, or html -> has question/answers
  - to store ex: FAQ
- create


4. fullfilment & integration
- fullfilment -> connect with Flask (backend)
- integration -> deploy..


### rapid api
1. create account

2. find suitable free api with less latency
- choose latency as 100ms

3. test copy token

4. used in flask webApp


=========Webhook========
https://cloud.google.com/dialogflow/docs/fulfillment-webhook
- It must handle HTTPS requests
https://cloud.google.com/serverless
https://cloud.google.com/products/compute

- must handle POST requests with a JSON WebhookRequest body.
- must respond to WebhookRequest requests with a JSON WebhookResponse body.

-----------
### context -> connect intent
https://www.youtube.com/watch?v=vRQ9RrGjQis
- teaching chatbot state
- using context -> to trigger chatbot intent

- output context (intent1) -> awaiting_name
- input context (intent2) -> awaiting_name
- actions -> in value -> use $name
- thanks $name

- what's your email, #awaiting_name.name?

### dialogflow techniques
https://www.youtube.com/watch?v=8g3aVYPY7d4&list=PLJLSPq0cTRma_kxRrNSAxcQKUxARMf2Vf

#### slot filling

- gettin user's contact data


============
### entities

- id ->

RE2 (google regex)
https://github.com/google/re2/wiki/Syntax
.
[xyz]
[^xyz]
xy -> x followed by y
x|y -> x or y

x* -> zero or more x
x+ -> 1 or more x
x? -> zero or 1 x
x{n,m} -> repition
x{n,} -> n or more
x{n} -> exactly repeated x n times.
x*? -> zero or more x

grouping :
(re) -> capturing group

i -> case insensitive

^ -> beginning string
$ 0> end text


[\d]	digits (≡ \d)
[^\d]	not digits (≡ \D)
\d	digits (≡ [0-9])
[[:digit:]]	digits (≡ [0-9])




https://cloud.google.com/dialogflow/docs/entities-overview
===
Entity type: Defines the type of information you want to extract from user input.







======
https://cloud.google.com/dialogflow/docs/integrations/telegram

custom payload
{
  "telegram": {
    "text": "You can read about *entities* [here](/docs/concept-entities).",
    "parse_mode": "Markdown"
  }
}
https://core.telegram.org/bots/api#formatting-options
