# seeGPT
analyze your AI usage by seeing what you use AI for the most and how you use AI over time  

currently have:
- cluster conversations to see how you have used at a point in time
- see you many messages you send over time (day, week, month, year)

will have:
- see what kind of messages you send over time
- hierarchical clusters
- update online

some concerns:  
- HDBSCAN classifies too many messsages as noise
- KMeans doesn't classify well/LLM doesn't name clusters well
- LLM naming won't work when clusters are super big

ideas:
- instead of machine finding clusters, first cluster according to intent categories then classify by topic
    - intent categories: asking, doing, expressing

useful resources:
- https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf