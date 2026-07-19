INSTRUCTIONS_PROMPT = """
You are an expert NLP data annotator.
Generate a batch of {batch_size} highly diverse and realistic sentences containing mentions of mountains, 
peaks, hills, or mountain ranges. 

The dataset style should match real-world natural text, including:
1. Short personal updates (e.g., "Just reached the summit of Mt. Rainier!").
2. Standard geographic facts / encyclopedic sentences (e.g., "The Andes Mountains extend across South America.").
3. Outdoor recreation and hiking blog snippets (e.g., "When hiking through the Carpathian Mountains, make sure 
to bring extra water.").

For each sentence, you must provide:
1. The raw sentence `text`.
2. A list of character `spans` marking the EXACT start and end index (0-indexed, start-inclusive, end-exclusive) of 
each mountain name entity within that exact `text`.
3. The raw `entity_names` corresponding to those spans.

Make sure to include some sentences with multiple mountain names and others with single names. 
Ensure absolute accuracy of the spans. Double-check your character counting. For example, in "I climbed Mt Fuji.", 
"Mt Fuji" starts at index 10 and ends at 17, so the span is [10, 17]. Ensure no off-by-one errors.
"""