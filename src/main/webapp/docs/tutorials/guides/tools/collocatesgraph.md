# Collocates Graph
	
Collocates Graph represents keywords and terms that occur in close proximity as a force directed network graph.

Use it with a <a href="../?view=CollocatesGraph&corpus=austen" target="_blank">Jane Austen corpus</a> or with <a href="../?view=CollocatesGraph" target="_blank">your own corpus</a>.

## Overview

This represents a network graph where keywords in green are shown linked to collocates in maroon. You can hover over a term to see its frequency (for keywords it's the corpus frequency, for collocates it's the frequeny in the context of the linked keywords). 

You can drag and drop terms to move them. You can drag terms off the canvas to remove them.

<iframe src="../tool/CollocatesGraph/?corpus=austen&subtitle=The+Works+of+Jane+Austen" style="width: 90%; height: 400px;"></iframe>
<div style="width: 90%; text-align: center; margin-bottom: 1em;">Collocates Graph with the Works of Jane Austen. You can also <a href="../?view=CollocatesGraph" target="_blank">use Collocates Graph with your own corpus</a>.</div>


## Options

You can add keywords by typing a query into the search box and hitting enter (see [Term Searches](#!/guide/search) for more advanced searching capabilities).

You can use the _Clear_ button to clear all keywords in the graph (to start from scratch and add your own).

The _Context_ slider determines how many terms to include when looking for collocates. The value specifies the number of words to consider on _each_ side of the keyword (so the total window of words is double). By default the context is set to 5 words per side, and the slider can have a maximum of 30.

Clicking on the [Options](#!/guide/options) icon also allows you to define a set of stopwords to exclude – see the [stopwords guide](#!/guide/stopwords) for more information.


## Spyral

To use Collocates Graph in Spyral you can use the following code as a starting point. Modify the config object to modify 
the visualization.

```
let config = {
    centralize: null, // If specified, will "centralize" on this keyword
    context: 5, // The number of terms to consider on each side of the keyword.
    limit: 5, // The number of items to limit the result to.
    query: null, // A query or array of queries (queries can be separated by a comma).
    stopList: "auto", // A comma-separated list of words, a named list or a URL to a plain text list, one word per line. By default this is set to 'auto' which auto-detects the document's language and loads an appropriate list (if available for that language). Set this to blank to not use the default stopList.
};

loadCorpus("austen").tool("collocatesgraph", config);
```

Please see {@link Tools.CollocatesGraph} for more information about configuration.



## See Also

- [Getting Started](#!/guide/start)
- [Stopwords](#!/guide/stopwords)
- [Term Searches](#!/guide/search)
- [Default Skin](#!/guide/skins-section-default-skin)
- [List of Tools](#!/guide/tools)
