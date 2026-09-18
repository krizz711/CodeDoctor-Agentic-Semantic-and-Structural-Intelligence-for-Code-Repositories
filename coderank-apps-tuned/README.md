---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:5000
- loss:MultipleNegativesRankingLoss
base_model: nomic-ai/CodeRankEmbed
widget:
- source_sentence: "Represent this query for searching relevant code: Your task is\
    \ to return the number of visible dots on a die after it has been rolled(that\
    \ means the total count of dots that would not be touching the table when rolled)\n\
    \n6, 8, 10, 12, 20 sided dice are the possible inputs for \"numOfSides\" \n\n\
    topNum is equal to the number that is on top, or the number that was rolled.\n\
    \nfor this exercise, all opposite faces add up to be 1 more than the total amount\
    \ of sides\n  Example: 6 sided die would have 6 opposite 1, 4 opposite 3, and\
    \ so on.\nfor this exercise, the 10 sided die starts at 1 and ends on 10.\n\n\
    Note: topNum will never be greater than numOfSides"
  sentences:
  - "from math import sqrt\ndef length_of_line(array):\n   x1, y1, x2, y2 = array[0][0],\
    \ array[0][1], array[1][0], array[1][1]\n   return '{:.2f}'.format((sqrt((x2-x1)**2\
    \ + (y2-y1)**2)))"
  - "def totalAmountVisible(topNum, numOfSides):\n    return numOfSides*(numOfSides+1)/2-(numOfSides-topNum+1)\n"
  - "class Solution:\n    def numDupDigitsAtMostN(self, N: int) -> int:\n        #\
    \ N -> total numbers less than or equal to N\n        # We will calculate the\
    \ integers with all different digits (which are less than/equal to N)\n      \
    \  # Then the answer would be: N - nums_with_different_digits\n        \n    \
    \    # Calculate the number of digits in N\n        NN, dd = N, 0\n        nums\
    \ = [] # store the digits\n        while(NN):\n            dd += 1\n         \
    \   nums.append(NN % 10)\n            NN //= 10\n        nums.reverse()\n    \
    \    \n        \n        # numbers with less digits than that of N\n        numbers\
    \ = 0\n        for i in range(dd-1):\n            numbers += 9 * (math.factorial(9)\
    \ // math.factorial(9-i))\n        \n        \n        # find the N-digit numbers\
    \ (all-different)\n        already_visited_digits = set()\n        \n        def\
    \ fac2(n, k):\n            return math.factorial(n) // math.factorial(n-k)\n \
    \       \n        for i,n in enumerate(nums):\n            k = 0\n           \
    \ for j in range((1 if i==0 else 0), (n+1 if i==dd-1 else n)):\n             \
    \   if(j in already_visited_digits):\n                    continue\n         \
    \       k += 1\n            numbers += k * fac2(10-i-1, dd-i-1)\n            if\
    \ n in already_visited_digits:\n                # All the numbers with this prefix\
    \ will have at least one common digit\n                break\n            already_visited_digits.add(n)\n\
    \            \n        return N - numbers"
- source_sentence: 'Represent this query for searching relevant code: Pavel made a
    photo of his favourite stars in the sky. His camera takes a photo of all points
    of the sky that belong to some rectangle with sides parallel to the coordinate
    axes.


    Strictly speaking, it makes a photo of all points with coordinates $(x, y)$, such
    that $x_1 \leq x \leq x_2$ and $y_1 \leq y \leq y_2$, where $(x_1, y_1)$ and $(x_2,
    y_2)$ are coordinates of the left bottom and the right top corners of the rectangle
    being photographed. The area of this rectangle can be zero.


    After taking the photo, Pavel wrote down coordinates of $n$ of his favourite stars
    which appeared in the photo. These points are not necessarily distinct, there
    can be multiple stars in the same point of the sky.


    Pavel has lost his camera recently and wants to buy a similar one. Specifically,
    he wants to know the dimensions of the photo he took earlier. Unfortunately, the
    photo is also lost. His notes are also of not much help; numbers are written in
    random order all over his notepad, so it''s impossible to tell which numbers specify
    coordinates of which points.


    Pavel asked you to help him to determine what are the possible dimensions of the
    photo according to his notes. As there are multiple possible answers, find the
    dimensions with the minimal possible area of the rectangle.



    -----Input-----


    The first line of the input contains an only integer $n$ ($1 \leq n \leq 100\,000$),
    the number of points in Pavel''s records.


    The second line contains $2 \cdot n$ integers $a_1$, $a_2$, ..., $a_{2 \cdot n}$
    ($1 \leq a_i \leq 10^9$), coordinates, written by Pavel in some order.



    -----Output-----


    Print the only integer, the minimal area of the rectangle which could have contained
    all points from Pavel''s records.



    -----Examples-----

    Input

    4

    4 1 3 2 3 2 1 3


    Output

    1

    Input

    3

    5 8 5 5 7 5


    Output

    0



    -----Note-----


    In the first sample stars in Pavel''s records can be $(1, 3)$, $(1, 3)$, $(2,
    3)$, $(2, 4)$. In this case, the minimal area of the rectangle, which contains
    all these points is $1$ (rectangle with corners at $(1, 3)$ and $(2, 4)$).'
  sentences:
  - "import re;\ndef check_password(s):\n    if re.search('^(?=.*?[a-z])(?=.*?[A-Z])(?=.*?\\\
    d)(?=.*?[!@#$%^&*?])[a-zA-Z\\d!@#$%^&*?]{8,20}$', s) :\n        return 'valid'\n\
    \    else:\n        return 'not valid'"
  - "def max_multiple(divisor, bound):\n    return bound - (bound % divisor)"
  - "#      \nimport collections, atexit, math, sys\nfrom functools import cmp_to_key\n\
    #key=cmp_to_key(lambda x,y: 1 if x not in y else -1 )\n\nsys.setrecursionlimit(1000000)\n\
    def getIntList():\n    return list(map(int, input().split()))    \n\nimport bisect\
    \ \ntry :\n    #raise ModuleNotFoundError\n    import numpy\n    def dprint(*args,\
    \ **kwargs):\n        print(*args, **kwargs, file=sys.stderr)\n    dprint('debug\
    \ mode')\nexcept ModuleNotFoundError:\n    def dprint(*args, **kwargs):\n    \
    \    pass\n\n\ndef memo(func):  \n    cache={}  \n    def wrap(*args):  \n   \
    \     if args not in cache:  \n            cache[args]=func(*args)  \n       \
    \ return cache[args]  \n    return wrap\n\n@memo\ndef comb (n,k):\n    if k==0:\
    \ return 1\n    if n==k: return 1\n    return comb(n-1,k-1) + comb(n-1,k)\n\n\
    inId = 0\noutId = 0\nif inId>0:\n    dprint('use input', inId)\n    sys.stdin\
    \ = open('input'+ str(inId) + '.txt', 'r') #标准输出重定向至文件\nif outId>0:\n    dprint('use\
    \ output', outId)\n    sys.stdout = open('stdout'+ str(outId) + '.txt', 'w') #标准输出重定向至文件\n\
    \    atexit.register(lambda :sys.stdout.close())     #idle 中不会执行 atexit\n    \n\
    N, = getIntList()\n\nzz = getIntList()\nzz.sort()\nresult = (zz[N-1] - zz[0])\
    \ * (zz[2*N-1] - zz[N])\n\ndif = zz[2*N-1] - zz[0]\n\nfor i in range(N):\n   \
    \ nd = zz[i+N-1] - zz[i]\n    result = min(result, nd*dif)\n\nprint(result)\n\n\
    \n\n\n\n\n\n"
- source_sentence: "Represent this query for searching relevant code: Kate constantly\
    \ finds herself in some kind of a maze. Help her to find a way out!.\n\nFor a\
    \ given maze and Kate's position find if there is a way out. Your function should\
    \ return True or False.\n\nEach maze is defined as a list of strings, where each\
    \ char stays for a single maze \"cell\". ' ' (space) can be stepped on, '#' means\
    \ wall and 'k' stays for Kate's starting position. Note that the maze may not\
    \ always be square or even rectangular.\n\nKate can move left, up, right or down\
    \ only.\n\nThere should be only one Kate in a maze. In any other case you have\
    \ to throw an exception.\n\n\nExample input\n=============\n```\n['# ##',\n '#\
    \ k#',\n '####']\n```\n\nExample output\n==============\nTrue\n\nExample input\n\
    =============\n```\n['####'.\n '# k#',\n '####']\n```\n\nExample output\n==============\n\
    False"
  sentences:
  - "n, m = list(map(int, input().split()))\ng = [[] for _ in range(n)]\nfor i in\
    \ range(m):\n    p, q = list(map(int, input().split()))\n    g[p - 1].append(q\
    \ - 1)\n    g[q - 1].append(p - 1)\n\ncomp = [-1] * n\ndef shortest(root):\n \
    \   dist = [-1] * n\n    q = [0] * n\n    left, right = 0, 1\n    q[left] = root\n\
    \    dist[root] = 0\n    good = True\n    while left < right:\n        x = q[left]\n\
    \        left = left + 1\n        for i in g[x]:\n            if dist[i] is -1:\
    \ \n                dist[i] = 1 + dist[x]\n                q[right] = i\n    \
    \            right = right + 1\n            elif dist[i] == dist[x]:\n       \
    \         good = False \n    far = 0\n    for i in dist: \n        if far < i:\n\
    \            far = i\n    return good, far, dist\n\narr = [0] * n\ngood = True\n\
    for i in range(n):\n    _, opt, dist = shortest(i)\n    if _ is False: good =\
    \ False\n    if comp[i] is -1:\n        for j in range(n): \n            if dist[j]\
    \ is not -1: comp[j] = i\n    if arr[comp[i]] < opt: \n        arr[comp[i]] =\
    \ opt\n\nif good is False: print('-1')\nelse: print(sum(arr))\n\n"
  - "MOVES = {(0,1), (0,-1), (1,0), (-1,0)}\n\ndef has_exit(maze):\n    posSet = {(x,y)\
    \ for x in range(len(maze)) for y in range(len(maze[x])) if maze[x][y] == 'k'}\n\
    \    if len(posSet) != 1:\n        raise ValueError(\"There shouldn't be more\
    \ than one kate\")\n    \n    seen = set(posSet)\n    while posSet:\n        x,y\
    \ = posSet.pop()\n        if any(not (0 <= x+dx < len(maze) and 0 <= y+dy < len(maze[x+dx]))\
    \ for dx,dy in MOVES):\n            return True\n        neighbors = {(x+dx, y+dy)\
    \ for dx,dy in MOVES if 0 <= x+dx < len(maze) and 0 <= y+dy < len(maze[x+dx])\n\
    \                                                        and maze[x+dx][y+dy]\
    \ == ' '\n                                                        and (x+dx, y+dy)\
    \ not in seen}\n        posSet |= neighbors\n        seen   |= neighbors\n   \
    \ return False"
  - "def sc(strng):\n    seen = set(strng)\n    return ''.join(a for a in strng if\
    \ a.swapcase() in seen)\n"
- source_sentence: "Represent this query for searching relevant code: # Task\n\n**_Given_**\
    \ an *array/list [] of n integers* , *find maximum triplet sum in the array* **_Without\
    \ duplications_**   .\n___\n\n# Notes :\n\n* **_Array/list_** size is *at least\
    \ 3* .\n\n* **_Array/list_** numbers could be a *mixture of positives , negatives\
    \ and zeros* .\n\n* **_Repetition_** of numbers in *the array/list could occur*\
    \ , So **_(duplications are not included when summing)_**.\n___\n# Input >> Output\
    \ Examples \n\n## **_Explanation_**:\n\n* As the **_triplet_** that *maximize\
    \ the sum* **_{6,8,3}_** in order , **_their sum is (17)_** \n\n* *Note* : **_duplications_**\
    \ *are not included when summing* , **(i.e) the numbers added only once** .\n\
    ___\n## **_Explanation_**:\n\n* As the **_triplet_** that *maximize the sum* **_{8,\
    \ 6, 4}_** in order , **_their sum is (18)_** , \n* *Note* : **_duplications_**\
    \ *are not included when summing* , **(i.e) the numbers added only once** .\n\
    ___\n## **_Explanation_**:\n\n* As the **_triplet_** that *maximize the sum* **_{12\
    \ , 29 , 0}_** in order , **_their sum is (41)_** , \n* *Note* : **_duplications_**\
    \ *are not included when summing* , **(i.e) the numbers added only once** .\n\
    ___\n# [Playing with Numbers Series](https://www.codewars.com/collections/playing-with-numbers)\n\
    \n# [Playing With Lists/Arrays Series](https://www.codewars.com/collections/playing-with-lists-slash-arrays)\n\
    \n# [For More Enjoyable Katas](http://www.codewars.com/users/MrZizoScream/authored)\n\
    ___\n___\n___\n\n## ALL translations are welcomed\n\n## Enjoy Learning !!\n# Zizou"
  sentences:
  - "import sys\ninput = sys.stdin.readline\nimport bisect\n\nt=int(input())\nfor\
    \ tests in range(t):\n    n=int(input())\n    A=list(map(int,input().split()))\n\
    \n    compression_dict={a: ind for ind, a in enumerate(sorted(set(A)))}\n    A=[compression_dict[a]\
    \ for a in A]\n\n    Q=[0]*n\n     \n    for i in range(n):\n        Q[A[i]]=i\n\
    \     \n    count=1\n    Ans=0\n    #print(A,Q)\n    for i in range(1,n):\n  \
    \      if Q[i]>Q[i-1]:\n            count+=1\n        else:\n            Ans=max(count,Ans)\n\
    \            count=1\n\n    Ans=max(count,Ans)\n     \n    print(n-Ans)\n\n  \
    \  \n"
  - "def two_highest(ls):\n    result = sorted(list(set(ls)), reverse=True)[:2]\n\
    \    return result if isinstance(ls, (list)) else False"
  - "def max_tri_sum(numbers):\n    return sum(sorted(set(numbers))[-3:])"
- source_sentence: 'Represent this query for searching relevant code: Given two strings
    s and t , write a function to determine if t is an anagram of s.


    Example 1:



    Input: s = "anagram", t = "nagaram"

    Output: true



    Example 2:



    Input: s = "rat", t = "car"

    Output: false



    Note:

    You may assume the string contains only lowercase alphabets.


    Follow up:

    What if the inputs contain unicode characters? How would you adapt your solution
    to such case?'
  sentences:
  - "t = int(input())\nfor _ in range(t):\n n = int(input())\n k = int(input())\n\
    \ num = int(k/n)\n x = max(n*(1+num) - k, 0)\n diff = abs(x - (n-x))\n if diff\
    \ == 0:\n  number = 2*x - 1\n else:\n  number = min(x, n-x)*2\n print(number)"
  - "from collections import defaultdict \ndef findSubarraySum(arr, n, Sum):  \n \n\
    \ # Dictionary to store number of subarrays  \n # starting from index zero having\
    \   \n # particular value of sum.  \n prevSum = defaultdict(lambda : 0) \n \n\
    \ res = 0 \n \n # Sum of elements so far.  \n currsum = 0 \n \n for i in range(0,\
    \ n):   \n \n  # Add current element to sum so far.  \n  currsum += arr[i] \n\
    \ \n  # If currsum is equal to desired sum,  \n  # then a new subarray is found.\
    \ So  \n  # increase count of subarrays.  \n  if currsum == Sum:   \n   res +=\
    \ 1 \n \n  # currsum exceeds given sum by currsum  - sum. \n  # Find number of\
    \ subarrays having   \n  # this sum and exclude those subarrays  \n  # from currsum\
    \ by increasing count by   \n  # same amount.  \n  if (currsum - Sum) in prevSum:\
    \ \n   res += prevSum[currsum - Sum] \n   \n \n  # Add currsum value to count\
    \ of   \n  # different values of sum.  \n  prevSum[currsum] += 1 \n  \n return\
    \ res \nn=int(input())\nlst=list(map(int,input().split()))\nif(n==1):\n if(lst[0]==0):\n\
    \  print(1)\n else:\n  print(0)\nelse:    \n print(findSubarraySum(lst,n,0)) \n"
  - "class Solution:\n     def isAnagram(self, s, t):\n         \"\"\"\n         :type\
    \ s: str\n         :type t: str\n         :rtype: bool\n         \"\"\"\n    \
    \     # dic = {}\n         # for item in s:\n         #     if item not in dic:\n\
    \         #         dic[item] = 1\n         #     else:\n         #         dic[item]\
    \ += 1\n         # for i in t:\n         #     if i not in dic:\n         #  \
    \       return False\n         #     else:\n         #         dic[i] -= 1\n \
    \        # return all(value == 0 for value in dic.values())\n         \n     \
    \    # fastest way till now with reference to the others' submissions\n      \
    \   if len(s) != len(t):\n             return False\n         if s == t:\n   \
    \          return True\n         for i in map(chr, range(97, 123)):\n        \
    \     if s.count(i) != t.count(i):\n                 return False\n         return\
    \ True"
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on nomic-ai/CodeRankEmbed

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [nomic-ai/CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [nomic-ai/CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) <!-- at revision 3c4b60807d71f79b43f3c4363786d9493691f8b1 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'NomicBertModel'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'cls', 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
queries = [
    'Represent this query for searching relevant code: Given two strings s and t\xa0, write a function to determine if t is an anagram of s.\n\nExample 1:\n\n\nInput: s = "anagram", t = "nagaram"\nOutput: true\n\n\nExample 2:\n\n\nInput: s = "rat", t = "car"\nOutput: false\n\n\nNote:\nYou may assume the string contains only lowercase alphabets.\n\nFollow up:\nWhat if the inputs contain unicode characters? How would you adapt your solution to such case?',
]
documents = [
    'class Solution:\n     def isAnagram(self, s, t):\n         """\n         :type s: str\n         :type t: str\n         :rtype: bool\n         """\n         # dic = {}\n         # for item in s:\n         #     if item not in dic:\n         #         dic[item] = 1\n         #     else:\n         #         dic[item] += 1\n         # for i in t:\n         #     if i not in dic:\n         #         return False\n         #     else:\n         #         dic[i] -= 1\n         # return all(value == 0 for value in dic.values())\n         \n         # fastest way till now with reference to the others\' submissions\n         if len(s) != len(t):\n             return False\n         if s == t:\n             return True\n         for i in map(chr, range(97, 123)):\n             if s.count(i) != t.count(i):\n                 return False\n         return True',
    'from collections import defaultdict \ndef findSubarraySum(arr, n, Sum):  \n \n # Dictionary to store number of subarrays  \n # starting from index zero having   \n # particular value of sum.  \n prevSum = defaultdict(lambda : 0) \n \n res = 0 \n \n # Sum of elements so far.  \n currsum = 0 \n \n for i in range(0, n):   \n \n  # Add current element to sum so far.  \n  currsum += arr[i] \n \n  # If currsum is equal to desired sum,  \n  # then a new subarray is found. So  \n  # increase count of subarrays.  \n  if currsum == Sum:   \n   res += 1 \n \n  # currsum exceeds given sum by currsum  - sum. \n  # Find number of subarrays having   \n  # this sum and exclude those subarrays  \n  # from currsum by increasing count by   \n  # same amount.  \n  if (currsum - Sum) in prevSum: \n   res += prevSum[currsum - Sum] \n   \n \n  # Add currsum value to count of   \n  # different values of sum.  \n  prevSum[currsum] += 1 \n  \n return res \nn=int(input())\nlst=list(map(int,input().split()))\nif(n==1):\n if(lst[0]==0):\n  print(1)\n else:\n  print(0)\nelse:    \n print(findSubarraySum(lst,n,0)) \n',
    't = int(input())\nfor _ in range(t):\n n = int(input())\n k = int(input())\n num = int(k/n)\n x = max(n*(1+num) - k, 0)\n diff = abs(x - (n-x))\n if diff == 0:\n  number = 2*x - 1\n else:\n  number = min(x, n-x)*2\n print(number)',
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 768] [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[ 0.8593, -0.0935, -0.0050]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 5,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                           | sentence_1                                                                           |
  |:---------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                               | string                                                                               |
  | modality | text                                                                                 | text                                                                                 |
  | details  | <ul><li>min: 49 tokens</li><li>mean: 339.47 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 147.08 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this query for searching relevant code: Complete the function to determine the number of bits required to convert integer `A` to integer `B` (where `A` and `B` >= 0)<br><br>The upper limit for `A` and `B` is 2^(16), `int.MaxValue` or similar.<br><br>For example, you can change 31 to 14 by flipping the 4th and 0th bit:<br>```<br> 31  0 0 0 1 1 1 1 1<br> 14  0 0 0 0 1 1 1 0<br>---  ---------------<br>bit  7 6 5 4 3 2 1 0<br>```<br>Thus `31` and `14` should return `2`.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | <code>def convert_bits(a,b):<br>    return bin(a^b).count("1")</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  | <code>Represent this query for searching relevant code: Create a function that interprets code in the esoteric language **Poohbear**<br><br>## The Language<br><br>Poohbear is a stack-based language largely inspired by Brainfuck. It has a maximum integer value of 255, and 30,000 cells. The original intention of Poohbear was to be able to send messages that would, to most, be completely indecipherable: <br>Poohbear Wiki<br><br>* For the purposes of this kata, you will make a version of Poohbear that has **infinite** memory cells in **both directions** (so you do not need to limit cells to 30,000)<br>* Cells have a default value of 0<br>* Each cell can hold one byte of data.  Once a cell's value goes above 255, it wraps around to 0.  If a cell's value goes below 0, it wraps to 255.<br>* If the result of an operation isn't an int, round the result down to the nearest one.<br>* Your interpreter should ignore any non-command characters in the code.<br>* If you come to a `W` in the code and the current cell is equal to 0, jump to the...</code> | <code>from operator import add, mul, floordiv as fdiv, pow<br><br>def poohbear(s):<br>    <br>    def updateMem(func, v): mem[p] = func(mem.get(p, 0), v) % 256<br><br>    braces, stack = {}, []<br>    for i,c in enumerate(s):<br>        if c == 'W': stack.append(i)<br>        if c == 'E':<br>            braces[i] = stack[-1]<br>            braces[stack.pop()] = i<br>    <br>    mem, copy, output = {}, 0, []<br>    p, i = 0, 0<br>    while i < len(s):<br>        cmd = s[i]<br>        if   cmd == '>':  p += 1<br>        elif cmd == '<':  p -= 1<br>        elif cmd == 'p':  mem[p] = copy<br>        elif cmd == 'c':  copy = mem.get(p, 0)<br>        elif cmd == 'W':  i = i if bool(mem.get(p, 0)) else braces[i]<br>        elif cmd == 'E':  i = braces[i] if mem.get(p, 0) else i<br>        elif cmd == 'P':  output.append(chr(mem.get(p, 0)))<br>        elif cmd == 'N':  output.append(str(mem.get(p, 0)))<br>        elif cmd == '+':  updateMem(add,  1)<br>        elif cmd == '-':  updateMem(add, -1)<br>        elif cmd == 'L':  updateMem(add,  2)<br>        elif cmd == 'I':  updateMem...</code> |
  | <code>Represent this query for searching relevant code: Your coworker was supposed to write a simple helper function to capitalize a string (that contains a single word) before they went on vacation.<br><br>Unfortunately, they have now left and the code they gave you doesn't work. Fix the helper function they wrote so that it works as intended (i.e. make the first character in the string "word" upper case).<br><br>Don't worry about numbers, special characters, or non-string types being passed to the function. The string lengths will be from 1 character up to 10 characters, but will never be empty.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>def capitalize_word(word):<br>    return word.capitalize()<br><br>    # .capitilize() for the first word in the string<br>    # .title() for each word in the string<br></code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 0.8   | 500  | 0.1308        |
| 1.6   | 1000 | 0.0513        |
| 2.4   | 1500 | 0.0248        |


### Training Time
- **Training**: 20.6 minutes

### Framework Versions
- Python: 3.13.15
- Sentence Transformers: 5.7.0
- Transformers: 5.16.1
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 4.8.5
- Tokenizers: 0.23.1

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->