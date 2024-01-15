Ext.data.JsonP.corpuscreator({"guide":"<h1 id='corpuscreator-section-creating-a-corpus'>Creating a Corpus</h1>\n<div class='toc'>\n<p><strong>Contents</strong></p>\n<ol>\n<li><a href='#!/guide/corpuscreator-section-sources'>Sources</a></li>\n<li><a href='#!/guide/corpuscreator-section-options'>Options</a></li>\n<li><a href='#!/guide/corpuscreator-section-input-format'>Input Format</a></li>\n<li><a href='#!/guide/corpuscreator-section-titles'>Titles</a></li>\n<li><a href='#!/guide/corpuscreator-section-documents-order'>Documents Order</a></li>\n<li><a href='#!/guide/corpuscreator-section-text'>Text</a></li>\n<li><a href='#!/guide/corpuscreator-section-xml'>XML</a></li>\n<li><a href='#!/guide/corpuscreator-section-html'>HTML</a></li>\n<li><a href='#!/guide/corpuscreator-section-json'>JSON</a></li>\n<li><a href='#!/guide/corpuscreator-section-tables'>Tables</a></li>\n<li><a href='#!/guide/corpuscreator-section-tokenization'>Tokenization</a></li>\n<li><a href='#!/guide/corpuscreator-section-access-management'>Access Management</a></li>\n<li><a href='#!/guide/corpuscreator-section-next-steps'>Next Steps</a></li>\n</ol>\n</div>\n\n<p>Voyant offers powerful functionality for creating your own corpus.</p>\n\n<h2 id='corpuscreator-section-sources'>Sources</h2>\n\n<p>The following sources are supported:</p>\n\n<ul>\n<li><strong>Text Box</strong>: you can type or paste text into the main text box in two different formats:\n\n<ul>\n<li>regular text as one document (plain text, HTML and XML are supported)</li>\n<li>a set of URLs, one per line</li>\n</ul>\n</li>\n<li><strong>Open</strong>: you can open an existing corpus that's already been created:\n\n<ul>\n<li><a href=\"../?corpus=austen\">Austen</a>: Project Gutenberg's collection of 8 novels from <a href=\"http://www.gutenberg.org/ebooks/author/68\">Jane Austen</a>: <em>Love And Freindship</em>, <em>Lady Susan</em>, <em>Sense and Sensibility</em>, <em>Pride and Prejudice</em>, <em>Mansfield Park</em>, <em>Emma</em>, <em>Northanger Abbey</em>, <em>Persuasion</em></li>\n<li><a href=\"../?corpus=shakespeare\">Shakespeare</a>: Project Gutenberg's collection of 37 plays from <a href=\"http://www.gutenberg.org/ebooks/author/65\">William Shakespeare</a></li>\n</ul>\n</li>\n<li><strong>Upload</strong>: you can upload one or more files from your computer\n\n<ul>\n<li>use Shift and Ctrl keys to select multiple files at once</li>\n<li>you can create a zip archive on your machine and upload it instead of selecting individual files</li>\n</ul>\n</li>\n</ul>\n\n\n<p>Unlike in the previous version of Voyant, you can now <a href=\"#!/guide/documents-section-modifying-a-corpus\">add, remove and reorder documents</a> after a corpus has been created.</p>\n\n<div style=\"max-width: 600px; margin-left: auto; margin-right: auto\"><p><img src=\"guides/corpuscreator/corpuscreator.png\" alt=\"Corpus Creator\" width=\"1747\" height=\"445\"></p></div>\n\n\n<h2 id='corpuscreator-section-options'>Options</h2>\n\n<p>Options should be specified before hitting the upload button or the reveal button.</p>\n\n<div style=\"max-width: 350px; margin-left: auto; margin-right: auto;\"><p><img src=\"guides/corpuscreator/options.png\" alt=\"Input Options\" width=\"871\" height=\"507\"></p></div>\n\n\n<h2 id='corpuscreator-section-input-format'>Input Format</h2>\n\n<p>Most document formats are fairly easy to detect automatically, Voyant does a decent job of extracting text from HTML, MS Word, MS Excel, ODT, Pages (Apple), PDF, plain text, RTF, XML, and others. You can also provide archives (.zip, .tar, .tgz, etc.) containing documents in those formats. If you want to specific a format (because auto-detect isn't working), you can select from the following:</p>\n\n<ul>\n<li><strong>Atom Syndication Format (<a href=\"https://en.wikipedia.org/wiki/Atom_(standard\">Atom</a>))</strong>: An XML-based format often used by news media, blogs, etc.</li>\n<li><strong>Dynamic Table of Contexts (DToC)</strong>: A specialized XML-based format used by the <a href=\"http://cwrc.ca/DToC_Documentation/\">Dynamic Table of Contexts project</a></li>\n<li><strong>Really Simple Syndication (<a href=\"https://en.wikipedia.org/wiki/RSS\">RSS</a>)</strong>: An XML-based format often used by news media, blogs, etc. Note that this is for RSS Version 2.0+, not RSS 1.0.</li>\n<li><strong>Text Encoding Initiative (<a href=\"http://www.tei-c.org/\">TEI</a>)</strong>: An XML-based format (essentially uses <code>//text</code> for content and <code>//teiHeader//title</code> and <code>//teiHeader//author</code> for metadata)</li>\n<li><strong>TEI Corpus</strong>: As above, except that produces multiple documents from <code>//TEI</code> tags</li>\n</ul>\n\n\n<h2 id='corpuscreator-section-titles'>Titles</h2>\n\n<p>It's possible to provide a title and subtitle for your corpus. These values aren't typically used in the\n<a href=\"#!/guide/skins-section-default-skin\">default skin</a> but are useful elsewhere, such as in the <a href=\"#!/guide/skins-section-dynamic-table-of-contexts-skin\">Dynamic Table of Contexts skin</a>.</p>\n\n<h2 id='corpuscreator-section-documents-order'>Documents Order</h2>\n\n<p>You can select an initial sort order for the documents. The default behaviour depends somewhat on the input format and other factors but is typically related to the alphabetic order of files or titles. You can also force the use of titles for sorting, or author or publication date metadata (when available).</p>\n\n<h2 id='corpuscreator-section-text'>Text</h2>\n\n<p>Voyant provides functionality for ignoring parts of documents, for instance to exclude a license statement from the text to be analyzed. The values specified are regular expressions (so care should be taken when using characters other than letters, since they may have special meaning).</p>\n\n<ul>\n<li>omit until: ignore everything until the start of this regular expression</li>\n<li>omit until after: ignore everything until the end of this regular expression</li>\n<li>omit from: ignore everything from the start of this regular expression onwards</li>\n<li>omit from after: ignore everything from the end of this regular expression onwards</li>\n</ul>\n\n\n<p>Here's a simple example of a document:</p>\n\n<pre><code>This is the license preceding the actual document.\nThe Document Title\nThe very interesting document text.\nThe license following the actual document.\n</code></pre>\n\n<p>In this simple case we can either use one of two combinations of values:</p>\n\n<ul>\n<li>omit until: <code>The Document Title</code></li>\n<li>omit from: <code>The license</code></li>\n</ul>\n\n\n<p>Or:</p>\n\n<ul>\n<li>omit until after: <code>the actual document.</code></li>\n<li>omit from after: <code>document text.</code></li>\n</ul>\n\n\n<h2 id='corpuscreator-section-xml'>XML</h2>\n\n<p>Voyant provides powerful functionality for creating a corpus from XML documents, in particular by using <a href=\"https://en.m.wikipedia.org/wiki/XPath\">XPath</a> expressions to define documents, content, and metadata like title and author.</p>\n\n<ul>\n<li><strong>Content</strong>: This allows you to use only a portion of a document as the body content (much like the body tag of an HTML document). Multiple nodes matching this XPath expression will be combined.</li>\n<li><strong>Title</strong>: This extracts the text only (no tags) from any matching nodes to be used as title metadata.</li>\n<li><strong>Author</strong>: This extracts the text only (no tags) from any matching nodes to be used as author metadata.</li>\n<li><strong>Documents</strong>: This allows you to extract multiple documents from an XML document (such as posts in an RSS feed, though usually selecting the RSS <em>Input Format</em> will do that automatically). When this is used in combination with the options below, the other XPath expressions will be relative to each sub-document (not to the original document root node).</li>\n<li><strong>Group by</strong>: When used in conjunction with a <em>Documents</em> option, this allows you to group multiple documents together that share the same XPath value. For instance, if you there are multiple &lt;speech&gt; documents, you can group all of the documents together based on the value of the speaker (so there would be one document per speaker with all of the speeches grouped together). This option is ignored if <em>Documents</em> isn't specified.</li>\n</ul>\n\n\n<p>Additional Metadata:</p>\n\n<ul>\n<li><strong>Publication Date</strong>: An indication of the publication date (there's no pre-defined format for this but often it's useful to have alphabetically sortable values such as a year number).</li>\n<li><strong>Publisher</strong>: The publisher of the document.</li>\n<li><strong>Location</strong>: The publication location of the document.</li>\n<li><strong>Keywords</strong>: Any keywords associated with the document.</li>\n<li><strong>Collection</strong>: An indication of the collection to which this document belongs.</li>\n</ul>\n\n\n<p>Finally, there's a box where you can provide user-defined metadata. This is currently used for advanced features in Voyant and won't be generally useful. The format is to have one entry per line where the metadata name points to an XPath expression:</p>\n\n<pre><code>xmlGenreXpath=//genre\nxmlVolumeXpath=//div/@volume\n</code></pre>\n\n<p>Both <em>Title</em> and <em>Author</em> XPath expressions can result in multiple values per document (some Voyant tools treat these separately and some combine them, depending on context). You of course have more control as needed:</p>\n\n<ul>\n<li><strong><code>//title</code></strong>: keep the text from every <code>title</code> tag as distinct values</li>\n<li><strong><code>(//title)[1]</code></strong>: keep only the text from the first <code>title</code> tag</li>\n<li><strong><code>//title|//h1</code></strong> keep the text from every <code>title</code> and <code>h1</code> tag as distinct values</li>\n<li><strong><code>//div[@type='chapter']/head</code></strong> keep the text from every <code>head</code> tag that's a child of a <code>div</code> tag whose type attribute is equal to <code>chapter</code></li>\n<li><strong><code>string-join(//author, '; ')</code></strong> combine all the text content from <code>author</code> tags into one value separated by <code>;</code></li>\n</ul>\n\n\n<p>This isn't the place to <a href=\"http://www.w3schools.com/xsl/xpath_syntax.asp\">learn XPath syntax</a>, but it's worth mentioning a few things about <a href=\"http://www.w3schools.com/xml/xml_namespaces.asp\">namespaces</a>. In most cases, you should be able to specify an XPath without any namespaces because Voyant will use the same default namespace as the document. You <em>can</em> use namespaces if you need to select elements in a given namespace. You can also create an XPath expression that only considers the local name of the element instead of the qualified name.</p>\n\n<ul>\n<li><strong><code>//creator</code></strong>: select the <code>creator</code> element using the default namespace</li>\n<li><strong><code>//dc:creator</code></strong>: select the <code>creator</code> element only when it is in the <code>dc</code> namespace</li>\n<li><strong><code>//*[local-name()='creator']</code></strong>: select any tag whose local name is <code>creator</code> regardless of namespace</li>\n</ul>\n\n\n<h2 id='corpuscreator-section-html'>HTML</h2>\n\n<p>New and <span style=\"color: red\">experimental</span>!</p>\n\n<p>You might want to work with only part of an HTML document, such as the main article without the rest of the navigation and other components on the page. You can now define CSS and jQuery type selectors that allow you to point to parts of an HTML document for the following:</p>\n\n<ul>\n<li><strong>Content</strong>: This defines the text content (by default it uses the HTML body tag). Multiple elements matching this query will be combined.</li>\n<li><strong>Title</strong>: This extracts the text only (no tags) from any matching elements to be used as title metadata.</li>\n<li><strong>Author</strong>: This extracts the text only (no tags) from any matching elements to be used as author metadata.</li>\n<li><strong>Documents</strong>: This allows you to extract multiple documents from an HTML document (such as individual posts in a blog). When this is used in combination with the options above, the other queries expressions will be relative to each sub-document (not to the original document root node).</li>\n<li><strong>Group by</strong>: When used in conjunction with a <em>Documents</em> option, this allows you to group multiple documents together that share the same value. For instance, if a page has multiple article tags, you can group all of the articles together based on the value of the author (so there would be one document per author with all of the articles from each grouped together). This option is ignored if <em>Documents</em> isn't specified.</li>\n</ul>\n\n\n<p>Additional Metadata:</p>\n\n<ul>\n<li><strong>Publication Date</strong>: An indication of the publication date (there's no pre-defined format for this but often it's useful to have alphabetically sortable values such as a year number).</li>\n<li><strong>Publisher</strong>: The publisher of the document.</li>\n<li><strong>Location</strong>: The publication location of the document.</li>\n<li><strong>Keywords</strong>: Any keywords associated with the document.</li>\n<li><strong>Collection</strong>: An indication of the collection to which this document belongs.</li>\n</ul>\n\n\n<p>Finally, there's a box where you can provide user-defined metadata. This is currently used for advanced features in Voyant and won't be generally useful. The format is to have one entry per line where the metadata name points to a selector:</p>\n\n<pre><code>htmlGenreQuery=.genre\nhtmlVolumeQuery=div@volume\n</code></pre>\n\n<p>These options currently use the <a href=\"https://jsoup.org/\">Jsoup library</a> and support its <a href=\"https://jsoup.org/cookbook/extracting-data/selector-syntax\">full syntax</a> as described below. Rather than using trial and error in creating Voyant corpora, you can first <a href=\"https://try.jsoup.org\" target=\"_blank\">try Jsoup</a> with your HTML document.</p>\n\n<h3 id='corpuscreator-section-selectors%3A'>Selectors:</h3>\n\n<ul>\n<li><strong>tagname</strong>: find elements by tag, e.g. <code>a</code></li>\n<li><strong>#id</strong>: find elements by ID, e.g. <code>#main</code></li>\n<li><strong>.class</strong>: find elements by class name, e.g. <code>.chapter</code></li>\n<li><strong>[attribute]</strong>: elements with attribute, e.g. <code>[role]</code></li>\n<li><strong>[^attr]</strong>: elements with an attribute name prefix, e.g. <code>[^data-]</code> finds elements with HTML5 dataset attributes</li>\n<li><strong>[attr=value]</strong>: elements with attribute value, e.g. <code>[role=main]</code> (also quotable, like <code>[role='main']</code>)</li>\n<li><strong>[attr^=value]</strong>: elements with attributes that start with the value, e.g. <code>[href^=http]</code></li>\n<li><strong>[attr^=value]</strong>: elements with attributes that start with the value, e.g. <code>[href*=voyant-tools]</code></li>\n<li><strong>[attr^=value]</strong>: elements with attributes that start with the value, e.g. <code>[href$=html]</code></li>\n<li><strong>[attr~=regex]</strong>: elements with attribute values that match the regular expression; e.g. `[href~=https?.+?(png|jpe?g)]</li>\n</ul>\n\n\n<h3 id='corpuscreator-section-selector-combinations'>Selector combinations</h3>\n\n<ul>\n<li><strong>el#id</strong>: elements with ID, e.g. <code>div#main</code></li>\n<li><strong>el.class</strong>: elements with class, e.g. <code>div.chapter</code></li>\n<li><strong>el[attr]</strong>: elements with attribute, e.g. <code>div[role]</code></li>\n<li><strong>ancestor child</strong>: child elements that descend from ancestor, e.g. <code>body p</code> finds p elements anywhere under a body element</li>\n<li><strong>parent > child</strong>: child elements that descend directly from parent, e.g. <code>article &gt; header &gt; h1</code> finds h1 elements whose immedidiate parents are header and immediate grand-parents are article</li>\n<li><strong>siblingA + siblingB</strong>: finds sibling B element immediately preceded by sibling A, e.g. <code>article &gt; header &gt; h1 + p</code> finds paragraphs that are siblings of the previous selector example</li>\n<li><strong>siblingA ~ siblingX</strong>: finds sibling X element preceded by sibling A, e.g. <code>h1 ~ p</code> finds h1 elements that are followed by p elements</li>\n<li><strong>el, el, el</strong>: group multiple selectors, find unique elements that match any of the selectors; e.g. <code>body &gt; h1, article header h1</code> find h1 elements that are immediate children of the body or that are descendences of article and header elements</li>\n</ul>\n\n\n<h3 id='corpuscreator-section-pseudo-selectors'>Pseudo selectors</h3>\n\n<ul>\n<li><strong>:lt(n)</strong>: find elements whose zero-based sibling index (i.e. its position in the DOM tree relative to its parent) is less than n; e.g. <code>td:lt(3)</code></li>\n<li><strong>:gt(n)</strong>: find elements whose zero-based sibling index is greater than n; e.g. <code>div p:gt(2)</code></li>\n<li><strong>:eq(n)</strong>: find elements whose zero-based sibling index is equal to n; e.g. form input:eq(0)</li>\n<li><strong>:has(selector)</strong>: find elements that contain elements matching the selector; e.g. <code>div:has(p)</code></li>\n<li><strong>:not(selector)</strong>: find elements that do not match the selector; e.g. <code>div:not(.navigation)</code></li>\n<li><strong>:contains(text)</strong>: find elements that contain the given text (in any descendant). The search is case-insensitive; e.g. <code>p:contains(text analysis)</code></li>\n<li><strong>:containsOwn(text)</strong>: find elements that directly contain the given text <code>p:contains(text analysis)</code></li>\n<li><strong>:matches(regex)</strong>: find elements whose text (in any descendant) matches the specified regular expression; e.g. <code>div:matches(19\\d\\d)</code></li>\n<li><strong>:matchesOwn(regex)</strong>: find elements whose own text matches the specified regular expression e.g. <code>div:matches(19\\d\\d)</code></li>\n</ul>\n\n\n<h3 id='corpuscreator-section-attribute-values'>Attribute Values</h3>\n\n<p>Selectors (as defined by the W3C and as implemented by Jsoup) are designed to select DOM elements or nodes. In many cases this is fine because simply want to use the text value of the element:</p>\n\n<pre><code>// for the \"Title\" field we could simply put \"title\" to select the title tag\n&lt;title&gt;Title&lt;/title&gt;\n</code></pre>\n\n<p>The problem is that in some cases we want the value of an attribute instead of the text content of an element. This isn't possible with selectors, but Voyant adds additional functionality by searching for an @ symbol and attribute name at the end of a selector, and if it's there, selecting the value</p>\n\n<pre><code>// for \"Author\" we could put \"meta[name='author']@content\"\n&lt;meta name=\"author\" content=\"Jane Austen\"&gt;\n</code></pre>\n\n<p>Note that in the case of grouped selectors, the same attribute applies to each group:</p>\n\n<pre><code>// only the value of author attribute is used: \".comment[author], header p[author] @author: \n&lt;div class=\"comment\" author=\"Jane Austen\"&gt;&lt;/div&gt;&lt;header&gt;&lt;p author=\"Jane Austen\"&gt;&lt;/p&gt;&lt;/header&gt;\n</code></pre>\n\n<p>The <a href=\"https://try.jsoup.org\" target=\"_blank\">try Jsoup</a> tool does NOT support this attribute functionality, but you can use it to ensure that you have the right elements and then add the @attributename as desired in Voyant.</p>\n\n<h2 id='corpuscreator-section-json'>JSON</h2>\n\n<p>New and <span style=\"color: red\">experimental</span>!</p>\n\n<p>You can now work with documents in JSON and use the <a href=\"https://tools.ietf.org/html/rfc6901\">JSON Pointer</a> syntax to select parts of the document, the same way you do with XML XPath or HTML CSS Selectors.</p>\n\n<p>The basic syntax is very simple, it somewhat resembles XPath but is less powerful. Every pointer should be a full address to the location in the document.</p>\n\n<p>Given an example JSON document like this:</p>\n\n<pre><code>{\n    \"rss\": {\n            \"items\": [{\n                \"title\": \"A Special Event\",\n                \"description\": \"A Special Teleconference for our customers about our products\",\n                \"link\": \"http://www.yourdomain.com/events.htm\",\n                \"author\": \"Joe Blow\"\n            },{\n                \"title\": \"Announcing new Products\",\n                \"description\": \"Announcing a new line of products\",\n                \"link\": \"http://www.yourdomain.com/events.htm\",\n                \"author\": \"Joe Blow\"\n            }]\n    }\n}\n</code></pre>\n\n<p>You could define the following:</p>\n\n<ul>\n<li><strong>Documents</strong>: /rss/items</li>\n<li><strong>Content</strong>: /description</li>\n<li><strong>Title</strong>: /title</li>\n<li><strong>Author</strong>: /author</li>\n</ul>\n\n\n<p>Notice how <strong>Documents</strong> is an array. Once defined, the author metadata pointers are relative to each document (not the root of the JSON document).</p>\n\n<h3 id='corpuscreator-section-pointers'>Pointers</h3>\n\n<ul>\n<li><strong>Content</strong>: This defines the text content (by default it uses the entire document but it's strongly recommended to define a pointer if applicable).</li>\n<li><strong>Title</strong>: This extracts the text to be used as title metadata.</li>\n<li><strong>Author</strong>: This extracts the text as author metadata (it can be an array or a string)</li>\n<li><strong>Documents</strong>: This allows you to extract multiple documents from an HTML document (such as individual posts in a blog). When this is used in combination with the options above, the other queries expressions will be relative to each sub-document (not to the original document root node).</li>\n</ul>\n\n\n<p>Additional Metadata:</p>\n\n<ul>\n<li><strong>Publication Date</strong>: An indication of the publication date (there's no pre-defined format for this but often it's useful to have alphabetically sortable values such as a year number).</li>\n<li><strong>Publisher</strong>: The publisher of the document.</li>\n<li><strong>Location</strong>: The publication location of the document.</li>\n<li><strong>Keywords</strong>: Any keywords associated with the document.</li>\n<li><strong>Collection</strong>: An indication of the collection to which this document belongs.</li>\n</ul>\n\n\n<p>Finally, there's a box where you can provide user-defined metadata. This is currently used for advanced features in Voyant and won't be generally useful. The format is to have one entry per line where the metadata name points to a selector:</p>\n\n<pre><code>htmlGenreQuery=/genre\nhtmlVolumeQuery=/doc/volume\n</code></pre>\n\n<p>If any Pointer is specified but doesn't exist in the document an exception will be raised and processing will fail. The error message should indicate which pointer failed.</p>\n\n<h2 id='corpuscreator-section-tables'>Tables</h2>\n\n<p>Voyant allows you to work flexibly with tabular data such as spreadsheets. At the moment the options described here only work with MS Excel files (.xsl or .xslx) and comma and tab-separated values files (CSV and TSV). Voyant can currently extract text from other tabular file formats such as OpenOffice and Pages, but in that case each file is considered as a separate document. The options below allow you to extract multiple documents from a single file (or from several files).</p>\n\n<p>The options for tables are a bit complex, but there are a lot of possibilities when working with tabular data, so it's worth it, right?</p>\n\n<p>The first option is for defining how Voyant should extract text from the table (file). There are three choices:</p>\n\n<p><b>1: <em>from entire table</em></b>: each table/file is considered one document, this is the default behaviour; only the <em>No Headers Row</em> option below is considered</p>\n\n<p><b>2: <em>from cells in each row</em></b>: this option assumes that each row has one or more documents, either the entire row or specific cells</p>\n\n<div style=\"max-width: 476px; margin-left: auto; margin-right: auto;\"><p><img src=\"guides/corpuscreator/fromcellsineachrow.png\" alt=\"From Cells in Each Row\" width=\"953\" height=\"266\"></p></div>\n\n\n<p><b>3: <em>from entire columns</em></b>: this option assumes that documents should be extracted from one or more columns</p>\n\n<div style=\"max-width: 371px; margin-left: auto; margin-right: auto;\"><p><img src=\"guides/corpuscreator/fromentirecolumns.png\" alt=\"From Entire Columns\" width=\"743\" height=\"253\"></p></div>\n\n\n<p>Whether you use <em>from cells in each row</em> or <em>from entire columns</em> you can also choose one or more columns for content. Columns are specified by number (even when there's a header row), and the left-most column is column 1. Content from multiple columns can be combined using the plus sign and columns can be specified separately by using commas. Here are some examples:</p>\n\n<ul>\n<li>1: use column one</li>\n<li>1,2: use columns one and two separately</li>\n<li>1+2,3 combine columns one and two and use column three separately</li>\n</ul>\n\n\n<p>When no <em>Content</em> value is specified the behaviour depends on the <em>Documents</em> option:</p>\n\n<ol>\n<li><em>from cells in each row</em>: each row is treated as a separate document (cells in a row are combined)</li>\n<li><em>from entire columns</em>: each column is treated as a separate document</li>\n</ol>\n\n\n<h4 id='corpuscreator-section-title-and-author-metadata'>Title and Author Metadata</h4>\n\n<p>The syntax is the same for the <em>Title</em> and <em>Author</em> options: column numbers separated by commas and/or combined with a plus sign (starting with the left-most column 1). These metadata options are only used when documents are extracted <em>from cells in each row</em> and when there's only one document per row (no <em>Content</em> value or no commas in the value of the <em>Content</em> option). When there's more than one document per row, a title label is automatically generated (no authors are defined).</p>\n\n<p>When documents are extracted <em>from entire columns</em>, the title metadata is extracted from the first row if there's a header row, otherwise a label is automatically generated (no authors are defined).</p>\n\n<h2 id='corpuscreator-section-tokenization'>Tokenization</h2>\n\n<p>Tokenization (in this context) is the process of identifying words, or sequences of Unicode letter characters that should be considered as a unit. In most cases Voyant will do a decent job of tokenization, even with some languages where there's not always an indication of word boundaries (like Chinese). There are three choices:</p>\n\n<ul>\n<li><strong>Automatic (highly recommended)</strong>: this works adequately for most languages</li>\n<li><strong>Simple Word Boundaries</strong>: use this if you have segmented the text yourself (by adding spaces between words), all non-letter characters (like punctuation) will be discarded from word tokens</li>\n<li><strong>Whitespace Only</strong>: use this if you want tokens to be created based solely on separation of whitespace (note that this may help for things like Twitter where you want @users and #hashes and urls.com, but it also means that all punctuation will remain attached to tokens)</li>\n</ul>\n\n\n<p>The following table summarizes tokenization for the string <span class=\"keyword\">What's voyant-tools.org?</span>:</p>\n\n<table class=\"grid rows\">\n<thead>\n<tr>\n<th> Tokenization </th>\n<th> Count </th>\n<th> Tokens </th>\n<th> Notes </th>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td> Automatic </td>\n<td style=\"text-align: center\"> 3 </td>\n<td> <span class=\"keyword\">what's</span>, <span class=\"keyword\">voyant</span>, <span class=\"keyword\">tools.org</span> </td>\n<td> the hyphen is split but the tools.org is considered a URL token; tokens are lowercase </td>\n</tr>\n<tr>\n<td> Word Boundaries </td>\n<td style=\"text-align: center\"> 5 </td>\n<td> <span class=\"keyword\">what</span>, <span class=\"keyword\">s</span>, <span class=\"keyword\">voyant</span>, <span class=\"keyword\">tools</span>, <span class=\"keyword\">org</span> </td>\n<td> any non-word character is a delimiter, tokens are lowercase </td>\n</tr>\n<tr>\n<td> Whitespace Only </td>\n<td style=\"text-align: center\"> 2 </td>\n<td> <span class=\"keyword\">What's</span>, <span class=\"keyword\">voyant-tools.org?</span> </td>\n<td> punctuation is kept in tokens and case is unchanged </td>\n</tr>\n</tbody>\n</table>\n\n\n<h2 id='corpuscreator-section-access-management'>Access Management</h2>\n\n<div style=\"max-width: 400px; float: right; padding: 1em;\"><p><img src=\"guides/corpuscreator/access-management.png\" alt=\"Access Management\" width=\"894\" height=\"511\"></p></div>\n\n\n<p>Voyant provides some basic access management functions that are intended to help control who can access a given corpus. It's worth mentioning that each corpus is given a unique 32-character code when it's created, which amounts to 2<sup>128</sup> or 340,282,366,920,938,463,463,374,607,431,768,211,456 possibilities. In other words, it's extremely unlikely that anyone would stumble upon your corpus by accident or by luck. That doesn't mean that your corpus is entirely safe from prying eyes, it's possible for a URL or for parameters to be detected during usual web communication, for instance.</p>\n\n<p>The access codes that can be specified in Voyant are an additional level of protection. These shouldn't be considered as passwords, not least because Voyant is not normally hosted on a secure server (with https traffic), so any access codes are transmitted in the clear. Still, under normal circumstances, the access codes can help further restrict access, if needed.</p>\n\n<p>If privacy and security are significant concerns for whatever reason (confidentiality of data, copyright, etc.), it's <em>strongly</em> recommended that you use a local, <a href=\"https://github.com/voyanttools/VoyantServer#voyant-server\">standalone version of Voyant</a> – it can even be used while offline (while not connected to the internet).</p>\n\n<p>Access management must be specified during corpus creation, it can't be specified once a corpus is already created (that's because it would be much more difficult to determine who created the corpus and therefore who can manage it).</p>\n\n<h3 id='corpuscreator-section-admin-codes'>Admin Codes</h3>\n\n<p>The first option allows you to specify one or more admin(istration) codes. Admin codes give you access to the corpus so that you can modify it (using the Modify feature in the Documents tool). If you don't specify admin codes, the access codes (if provided) will still be in effect. You can specify one or more different admin codes separated by commas (any one of the codes will work).</p>\n\n<h3 id='corpuscreator-section-access-codes'>Access Codes</h3>\n\n<p>The second option allows you to specify one or more full access codes (without a valid code, access is either restricted or completely blocked – see the next option for more details). You can specify multiple codes separated by commas, which allows you to  assign and modify access independently for multiple groups and users. If no access codes are provided, access will be limited either to admin codes (if any are provided) or the corpus will be open.</p>\n\n<h3 id='corpuscreator-section-other-access'>Other Access</h3>\n\n<p>This option determines what happens when an admin or access code is required but no valid code is provided by the user:</p>\n\n<ul>\n<li><strong>limited (non-consumptive)</strong>: users can access analytic tools and views of the corpus but not any tool that allows text to be read or reconstituted</li>\n<li><strong>none</strong>: no access is provided to this corpus</li>\n</ul>\n\n\n<p>Although it might be tempting to select \"none\" for simplicity or by force of habit, the non-consumptive option is more nuanced solution and recognizes that much analytic work can be done with derivative data while protecting principles of copyright (since the text in its original form can't be recovered with non-consumptive access). These issues have been explored by <a href=\"http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2102542\">digital humanities scholars</a>, as well as by the courts in cases like <a href=\"https://en.wikipedia.org/wiki/Authors_Guild,_Inc._v._HathiTrust\">Authors Guild v. HathiTrust</a>.</p>\n\n<p>We believe that the non-consumptive option is on firm ethical and legal footing, even for copyright text, but responsibilty lies with the creator of the corpus. It's also worth reiterating that any access management provided by Voyant is only one line of defense, so unintended access may occur and the hosted version should not be used when confidentiality is important.</p>\n\n<h2 id='corpuscreator-section-next-steps'>Next Steps</h2>\n\n<ul>\n<li><a href=\"#!/guide/modifyingcorpus\">modifying a corpus</a></li>\n<li><a href=\"#!/guide/tools\">explore the tools</a></li>\n<li>read <a href=\"#!/guide/about\">about Voyant Tools</a></li>\n</ul>\n\n","title":"Creating a Corpus"});