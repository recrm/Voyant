# Getting Started

Voyant Tools is a web-based text reading and analysis environment. It's designed to make it easy for you to work with 
your own text or collection of texts in a variety of formats, including plain text, HTML, XML, PDF, RTF, and MS Word.

![Getting Started](imgs/ui/start/getting-started.png)

## Selecting a Corpus

There are three main ways of selecting a corpus in Voyant Tools:

1. type or paste into the main text area, either normal text or a set URLs, one per line; then hit the "Reveal" button
1. open an existing corpus (such as [Austen](../?corpus=austen) or [Shakespeare](../?corpus=shakespeare)
1. upload one or more files from your computer

![Select Corpus](imgs/ui/start/select-corpus.png)

The upload file selector should allow you to choose one or more files using Ctrl and Shift keys. If you have several 
documents to add at once, it may be easiest to first create a zip archive containing the files and then upload the one 
zip file.

For more information on selecting a corpus, including advanced options, see [Loading a Corpus]{@tutorial loadingcorpus}

<!--- add link to video -->

## Working in the Default Skin

Once you select a corpus you will be presented with the default skin (the default configuration of tools) that includes the following tools:

1. [Cirrus]{@tutorial cirrus}
1. [Reader]{@tutorial reader}
1. [Trends]{@tutorial trends}
1. [Summary]{@tutorial summary}
1. [Contexts]{@tutorial contexts}

<iframe src="../?corpus=austen" style="width: 90%; height: 600px;"></iframe>
<div style="width: 90%; text-align: center; margin-bottom: 1em;">Voyant with the Works of Jane Austen. You can also <a href="../" target="_blank">use Voyant with your own corpus</a>.</div>


The various tools in the interface are designed to interact with one another. For instance, if you click on a word in 
[Cirrus]{@tutorial cirrus}, you’ll see the [Trends]{@tutorial trends} tool update with information about the selected 
work. Similarly, if you click on a node in the [Trends]{@tutorial trends} tool the [Contexts]{@tutorial contexts} 
tool should update as well. Interactivity and navigation between the different scales of a corpus (from the macroscopic 
[Cirrus]{@tutorial cirrus} overview to the microscopic individual word occurrences) are a key part of the design of 
Voyant Tools.

### Additional Tools

Additional tools are readily accessible by clicking the tabs in each tool pane. For instance, beside the 
[Cirrus]{@tutorial cirrus} header label is the [Corpus Terms]{@tutorial corpusterms} label, clicking on the tab will 
switch the tool. Tools readily available through the tabs are [Corpus Terms]{@tutorial corpusterms}, 
[Links]{@tutorial collocatesgraph}, [Collocates]{@tutorial corpuscollocates}, [Documents]{@tutorial documents}, 
[Phrases]{@tutorial phrases}, and [Bubblelines]{@tutorial bubblelines}.

Beyond the tools visible in the tabs, it's possible to access most of the other tools available by clicking on the 
window icon that appears when the mouse is in the header of a tool (grey) or of the skin (blue). The menu that appears 
shows recommended tools for that location above the line and a hierarchy of other tools below the line.

![More Tools](imgs/ui/start/more-tools.png)

For more information about the available tools, consult the [List of Tools]{@tutorial tools}

## Bookmarking Your Corpus

One of the most interesting features of Voyant Tools is the ability to bookmark and share URLs that refer to your 
collection of texts. Among other advantages, this allows you to work with the same texts during different sessions, 
without having to reload all the documents each time. You can export a link for your corpus and the current set of 
tools clicking on the “Export” (diskette) icon in the blue bar at the top, or export a link for an individual tool by 
clicking on the “Export” icon in one of the tool panes.

Generally speaking, a corpus will remain accessible as long as it accessed at least once a month.

![Export Bookmark](imgs/ui/start/save-bookmark.png)

## Next Steps

* [creating a corpus]{@tutorial corpuscreator}
* [explore the tools]{@tutorial tools}
* read [about Voyant Tools]{@tutorial about}