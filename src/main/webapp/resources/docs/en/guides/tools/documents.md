# Documents

The Documents tool shows a table of the documents in the corpus and includes functionality for modifying the corpus.

Use it with a [Jane Austen corpus](../?view=Documents&corpus=austen) or with [your own corpus](../?view=Documents).


## Overview

The table view shows the following columns by default:

- *Title*: the document's title (or its filename if no better title was found)
- *Words*: the number of individual words (tokens) found in the document (e.g. each occurrence of "the" is counted)
- *Types*: the number of word forms found in the document (e.g. all occurrences of "the" are counted as one word form)
- *Ratio*: the ratio of types to tokens (types/tokens), expressed as a percentage – higher numbers generally mean greater vocabulary diversity
- *Words/Sentence*: an approximation of the average number of words per sentence (words tokens / sentences count); the way that sentences are calculated should be considered very approximate, especially because of complications with abbreviations and other uses of punctuation (parsing of sentences is performed by [Java's BreakIterator](https://docs.oracle.com/javase/tutorial/i18n/text/about.html) class, and also depends on accurate language detection)

Additional columns can be shown:

- *Author*: the document's author (if it can be determined)
- *Language*: the document's language (if it can be guessed)

By default, documents are shown in the order they exist in the corpus.

<iframe src="../tool/Documents/?corpus=austen&subtitle=The+Works+of+Jane+Austen" style="width: 90%; height: 350px;"></iframe>
<div style="width: 90%; text-align: center; margin-bottom: 1em;">Documents with the Works of Jane Austen. You can also <a href="../?view=Documents" target="_blank">use Documents with your own corpus</a>.</div>


## Options

You can filter documents  by typing a query into the search box and hitting enter (see {@tutorial search} for more 
advanced searching capabilities). Note that by default the query includes the full-text, the title and the author. You 
can use the author or title prefix to *only* look at one of those metadata fields (e.g. _title:love_).

You can modify a corpus by clicking on the _Modify_ button – see more information about {@tutorial modifyingcorpus}.

## Spyral

To use Documents in Spyral you can use the following code as a starting point. Modify the config object to modify 
the visualization.

```
let config = {
    "columns": null,
    "dir": null,
    "docId": null,
    "docIndex": null,
    "query": null,
    "sort": null,
};

loadCorpus("austen").tool("documents", config);
```

Please see {@link Tools.Documents} for more information about configuration.

## Additional Information

The type/token _Ratio_ value can be a useful way of expressing vocabulary richness, but the value is somewhat sensitve 
to document length and should be considered with circumspection. A more reliable way of measuring vocabulary richness 
is to average the type/token ratios from equally long segments in a text (e.g. the mean of type/token ratios for each 
1,000 word segment in the text).

We hope to soon offer functionality for users to edit or customize the metadata for documents, allowing you to edit the 
author or title, for instance. In the meantime, these metadata are defined during {@tutorial corpuscreator}.

## See Also
- {@tutorial start}
- {@tutorial grids}
- {@tutorial skins}
- {@tutorial tools}
