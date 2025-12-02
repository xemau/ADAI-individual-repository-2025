### Question 1: Are LLMs actually the best AI to use within this use case?
I do not believe they are the best option. The student justifies using GenAI by claiming it can handle "irregular demand behavior" and "temporal patterns" better than traditional ML. He argues it can "extrapolate beyond existing data". But using a text-based model for numerical forecasting is inefficient. It forces the computer to treat numbers as language tokens rather than mathematical values.

### Question 2: Why overstocking is bad?
Overstocking is detrimental because it ties up "working capital" and occupies "valuable warehouse space". The student notes that this leads to "elevated holding costs" and reduces the company's flexibility in storage and procurement.

### Question 3: Why does the prompt look like this?
The prompt looks like a text sentence because LLMs are designed to process natural language. To make the model work, the student had to structure the data as a question: "What is the output for product... if June-October are...". This tricks the model into completing the "sentence" with a number.

### Question 4: What do the numbers actually represent? Why are they always negative?
The numbers represent the "output" or usage of semi-finished products. They are likely negative because they represent consumption or items leaving the inventory. The data sample shows months like Jan and Feb having values like "-147" and "-84".

### Question 5: What things did the marketing student not know (knowledge wise), that brought him to believe LLMs were the answers?
Paul lacked knowledge of standard Data Science and Time-Series Forecasting. He focuses heavily on "Generative AI" creating new content but applies it to a task that requires precise numerical prediction. He conflates general "Advanced AI" with "GenAI" and seemingly ignores that standard "Predictive AI" is the industry standard for this type of problem.

### Question 6: Would predictive AI work suit better in this use case?
Yes, Predictive AI would suit this much better. Models designed for numerical regression, like Random Forests or specific Time-Series models, are built for this. The student admits his LLM is a "black box". Specialized predictive models like Prophet or Croston’s method would handle the "lumpy" demand he describes without the complexity of a language model.

### Question 7: What approach should we use with GenAi in order to make some tangible results in 4 weeks?
I would recommend a benchmarking approach. We should take the student's dataset and run a simple statistical algorithm (like a Moving Average) on it. If a simple calculation beats his "15% MAPE", we prove we do not need complex AI. We should simply test the one we think is best against his baseline rather than building a new tool from scratch.

### Question 8: Is the amount of data sufficient?
No, the data is likely insufficient. The student states the dataset spans "at least one full calendar year". One year is not enough to effectively show seasonal trends or avoid overfitting. You typically need multiple years of data to distinguish a true seasonal pattern from a random monthly fluctuation.

### Question 9: How many years of data we have?
We have only one year. The report specifies using historical output data for "only one year" to train and test the prototype.

### Question 10: Is AI the answer at all?
AI might not be the answer. The company currently uses a basic "linear" distribution of yearly forecasts. A "suited algorithm," such as a weighted moving average or basic regression, would likely improve performance dramatically over the current method without requiring any AI.

### Question 11: Is there any additional data that we can use to train the model on besides company data?
Yes, we could look for external drivers. The company sells aluminum hardware like "door handles, shields, and letter plates". The student suggests future research should include "macroeconomic indicators," "weather conditions," or "promotional activities". Data on the construction market or aluminum prices would likely be relevant.

### Question 12: What should the output look like?
The output should be numbers. The student had to specifically prompt the model to "only output a number" to ensure it provided a usable digit for the forecast. The business needs specific production volumes.

