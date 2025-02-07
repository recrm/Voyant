# Contexts

The Contexts (or Keywords in Context) tool shows each occurrence of a keyword with a bit of surrounding text 
(the context). It can be useful for studying more closely how terms are used in different contexts.

Use it with a [Jane Austen corpus](../?view=Contexts&corpus=austen) or with [your own corpus](../?view=Contexts).

## Overview

The table view shows the following three columns by default:

- *Document*: this is document in which keyword and contexts occur
- *Left*: contextual words to the left of the keyword (note that sorting by this column treats words in reverse order, right to left from the keyword)
- *Term*: the keyword matching the default or user-provided term query
- *Right*: contextual words to the right of the keyword

An additional column can be shown to display the term *Position* (token index) in the document.

By default, contexts are shown for the most frequent in the term corpus.

Some context is shown for each occurrence, you can also click on the plus icon to expand any given row to show more context.

<iframe src="../tool/Contexts/?corpus=austen&subtitle=The+Works+of+Jane+Austen" style="width: 90%; height: 350px;"></iframe>
<div style="width: 90%; text-align: center; margin-bottom: 1em;">Contexts with the Works of Jane Austen. You can also <a href="../?view=Contexts" target="_blank">use Contexts with your own corpus</a>.</div>

## Options

You can specify which keyword to use by typing a query into the search box and hitting enter (see {@tutorial search} 
for more advanced searching capabilities).

There is also a slider that determines how much context to consider when looking for collocates. The value specifies 
the number of words to consider on _each_ side of the keyword (so the total window of words is double). By default the 
context is set to 5 words per side, and the slider can have a maximum of 50. Similarly, there's an "expand" slider 
which determines how many words to show when you expand any given row (by clicking the plus icon in the left-most 
column). The default is 50, the minimum is 5 and the maximum is 500.

Clicking on the {@tutorial options} icon also allows you to define a set of stopwords to exclude – see the 
{@tutorial stopwords} for more information.

## Spyral

To use Catalogue in Spyral you can use the following code as a starting point. Modify the config object to modify 
the visualization.

```
let config = {
    columns: null, // 'docIndex', 'left', 'term', 'right', 'position'
    context: 5, // The number of terms to consider on each side of the keyword.
    dir: null, // The direction in which to sort the results: 'asc' or 'desc'
    docId: null, // The document ID(s) to restrict the results to.
    docIndex: null, // The document index(es) to restrict the results to.
    expand: null, // How many terms to show when you expand any given row
    query: null, // A query or array of queries (queries can be separated by a comma).
    sort: null, // The column to sort the results by
    stopList: null, // A comma-separated list of words, a named list or a URL to a plain text list, one word per line. By default this is set to 'auto' which auto-detects the document's language and loads an appropriate list (if available for that language). Set this to blank to not use the default stopList.
    termColors: null // Which term colors to show in the grid. By default this is set to 'categories' which shows the term color only if it's been assigned by a category. The other alternatives are 'terms' which shows all terms colors, and '' or undefined which shows no term colors.
};

loadCorpus("austen").tool("Contexts", config);
```

Please see {@link Tools.Contexts} for more information about configuration.

## Additional Information

For a graphical view of corpus collocates, try the {@tutorial collocatesgraph} tool.

## See Also
- {@tutorial start}
- {@tutorial grids}
- {@tutorial skins}
- {@tutorial tools}
