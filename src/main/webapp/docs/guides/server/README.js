Ext.data.JsonP.server({"guide":"<h1 id='server-section-voyantserver'>VoyantServer</h1>\n<div class='toc'>\n<p><strong>Contents</strong></p>\n<ol>\n<li><a href='#!/guide/server-section-setting-up-voyantserver'>Setting up VoyantServer</a></li>\n<li><a href='#!/guide/server-section-running-voyantserver'>Running VoyantServer</a></li>\n<li><a href='#!/guide/server-section-controlling-voyantserver'>Controlling VoyantServer</a></li>\n<li><a href='#!/guide/server-section-troubleshooting-voyantserver'>Troubleshooting VoyantServer</a></li>\n</ol>\n</div>\n\n<p>VoyantServer is a standalone version of the Voyant Tools server that can be downloaded and run locally. This standalone version allows you to do your text analysis on your computer without depending on any internet connection. Thus, it means:</p>\n\n<ul>\n<li>You can keep your texts confidential as our server will not cache your work.</li>\n<li>You can restart the server if it slows down or crashes.</li>\n<li>You can handle large texts without the connection timing out.</li>\n<li>You can work offline (without an Internet connection).</li>\n<li>You can have a group of users (like in a workshop or class) run their instance without encountering load issues on our server.</li>\n</ul>\n\n\n<p>This document outlines how to download, run, and control the server.</p>\n\n<h2 id='server-section-setting-up-voyantserver'>Setting up VoyantServer</h2>\n\n<h3 id='server-section-requirements'>Requirements</h3>\n\n<p>Ensure your computer has Java version 8 or the latest. Instructions for installing Java are available here -- <a href=\"https://www.wikihow.com/Run-a-.Jar-Java-File\">https://www.wikihow.com/Run-a-.Jar-Java-File</a>. Here is another link for how to install Java -- <a href=\"https://java.com/en/download/help/download_options.html\">https://java.com/en/download/help/download_options.html</a></p>\n\n<h3 id='server-section-downloading-voyant-server'>Downloading Voyant Server</h3>\n\n<ol>\n<li>Go to <a href=\"https://github.com/voyanttools/VoyantServer/releases/latest\">https://github.com/voyanttools/VoyantServer/releases/latest</a></li>\n<li>Scroll down to the bottom and click on the <em>VoyantServer.zip</em> to download.</li>\n</ol>\n\n\n<h3 id='server-section-preparing-voyantserver'>Preparing VoyantServer</h3>\n\n<p>First, go to the folder to which you downloaded <em>VoyantServer.zip</em>. The procedure for preparing VoyantServer is dependent on your computer's operating system (OS). Please see below and follow instructions for your respective OS.</p>\n\n<ol>\n<li><strong>Mac:</strong> You can double-click on the downloaded <em>VoyantServer.zip</em> file to decompress it. This should create a directory in the same folder as the zip file with a similar name to the downloaded file.</li>\n<li><strong>Windows:</strong> Extract the <em>VoyantServer.zip</em> onto the computer. You can do the extraction by right-clicking on the <em>VoyantServer.zip</em> and then click on \"<em>Extract All...</em>\"</li>\n<li><strong>Linux:</strong> You can double-click on the <em>VoyantServer.zip</em> file to decompress it. This should create a directory in the same folder as the zip file with a similar name to the downloaded file.</li>\n</ol>\n\n\n<p>Once you decompress the .zip file, you should see something like the following:</p>\n\n<p><code>_app</code>: this is the actual web application – you shouldn't need to view this folder's contents</p>\n\n<p><code>License.txt</code>: this is the license for the VoyantServer</p>\n\n<p><code>META-INF</code>: this is a part of the VoyantServer architecture – you shouldn't need to view this folder's contents</p>\n\n<p><code>README.md</code>: this includes some of the same documentation as on this page</p>\n\n<p><code>server-settings.txt</code>: this is an advanced way to set server options, including the port and memory defaults</p>\n\n<p><code>VoyantServer.jar</code>: this is the most important file, the one you'll click to start the server</p>\n\n<h2 id='server-section-running-voyantserver'>Running VoyantServer</h2>\n\n<ol>\n<li>Open the folder you just decompressed or extracted.</li>\n<li><p>To start VoyantServer, you need to run the <code>VoyantServer.jar</code> file. This Java Archive file is a package with all the resources needed to run the server (including an embedded Jetty server). To run this, you need to <a href=\"http://www.wikihow.com/Run-a-.Jar-Java-File\">have Java installed</a>. Note that the approach for running <code>VoyantServer.jar</code> is OS-specific. Please see below for the method that suits your computer's OS.</p>\n\n<ol>\n<li><p><strong>Mac:</strong></p>\n\n<ol>\n<li>You should right-click (control-click) on the <code>VoyantServer.jar</code> file and choose \"<em>Open\"</em> from the menu.</li>\n<li>Click on \"<em>Open\"</em> in the next dialog. This is necessary the first time only. For subsequent uses of VoyantServer, all you need to do is double-click on the <code>VoyantServer.jar</code> file.\n<p><img src=\"guides/server/01.png\" alt=\"Open\" width=\"300\" height=\"253\"></p></li>\n<li>If the clicking on <code>VoyantServer.jar</code> fails, use the command-line approach below.</li>\n</ol>\n</li>\n<li><p><strong>Windows / Linux:</strong></p>\n\n<ol>\n<li>You should be able to run VoyantServer by double-clicking on the <code>VoyantServer.jar</code> file</li>\n</ol>\n</li>\n<li><p><strong>Command-line:</strong></p>\n\n<ol>\n<li>It should also be possible to launch the application from the command-line if you're at the prompt in the same folder as the jar file: <code>java -jar VoyantServer.jar</code></li>\n</ol>\n</li>\n</ol>\n</li>\n</ol>\n\n\n<p>Once you run VoyantServer, you will see a control panel like the images below:</p>\n\n<p><p><img src=\"guides/server/02.png\" alt=\"Control Panel\" width=\"1024\" height=\"691\"></p></p>\n\n<p>Typically, VoyantServer will also automatically launch your browser with the Voyant Tools home screen, where you can define a text and get started.</p>\n\n<p><p><img src=\"guides/server/03.png\" alt=\"Website\" width=\"1024\" height=\"788\"></p></p>\n\n<h2 id='server-section-controlling-voyantserver'>Controlling VoyantServer</h2>\n\n<p>From the VoyantServer control panel you can:</p>\n\n<p><p><img src=\"guides/server/04.png\" alt=\"Control Panel\" width=\"1214\" height=\"250\"></p></p>\n\n<p>(1)  <strong>Stop Server / Start Server:</strong>  This button's label will depend on the state of VoyantServer – it will be disabled if it is in the process of starting up the VoyantServer. It will say <em>Stop Server</em> if VoyantServer is already running and <em>Start Server</em> if it isn't. You can stop VoyantServer if it fails to work as expected and then restart it. Note: You should always use the <em>Stop Server</em> button to quit the program, as it helps to release allocated resources. Otherwise, re-launching VoyantServer may not work.</p>\n\n<p>(2) <strong>Open Web:</strong> You can open your default web browser with the Voyant Tools entry page that connects with this server. By default, the URL will be http://127.0.0.1:8888. You can always connect with a local server by typing this into the URL field of your browser if you want to use another browser.</p>\n\n<p>(3) <strong>File -> Exit:</strong> You can quit the VoyantServer application.</p>\n\n<p>(4) <strong>Help:</strong> You can access the help page for the VoyantServer from the Help menu.</p>\n\n<p>(5) <strong>Port:</strong> You can change the port that is used by the server (the default is port 8888). Typically, this won't need to be changed – it's not recommended to make changes here unless you need to and know what you're doing. If the port specified is already in use, you can try a slightly different one, 8889, for instance.</p>\n\n<p>(6) <strong>Memory:</strong> You can increase the memory (in megabytes) allocated to the VoyantServer if you are analyzing large texts. Make sure you stop and restart the server for the new memory setting to take effect. The default is 1024 (MB).</p>\n\n<h2 id='server-section-troubleshooting-voyantserver'>Troubleshooting VoyantServer</h2>\n\n<ol>\n<li><p>If <strong>VoyantServer fails to start</strong> by clicking on the <code>VoyantServer.jar</code> file, use the command-line approach below, or if you get an error message that looks like the image below, you can solve the problem by launching the application from the command-line.\n<p><img src=\"guides/server/05.png\" alt=\"Error\" width=\"426\" height=\"157\"></p></p>\n\n<ol>\n<li>Make sure you are in the same folder or directory as the <code>VoyantServer.jar</code> file.</li>\n<li>If you are on the command-line and in the same folder as the jar file, then copy and paste <code>java -jar VoyantServer.jar</code> into your command-line.</li>\n<li>Press \"Enter\" to open VoyantServer.</li>\n</ol>\n</li>\n<li><p><strong>Port in Use Error:</strong> If you get a message that reads: <code>Port#8888 appears to be in use already</code>. It means your VoyantServer did not exit properly. Try the following steps to resolve the issue.</p>\n\n<p> The server, in this case, is running. But it can be hard to stop and restart. One option is to change the port (to 8889) and launch another server. But each time you launch a new server, more memory is being used. Another option is to restart your computer – that will clear existing processes. If you know what you're doing, you can also use the command line to find the process and kill it:</p>\n\n<pre><code class=\"`\"> ps aux | grep jetty\n kill -9 [process id]\n</code></pre>\n\n<p> <p><img src=\"guides/server/06.png\" alt=\"Command Line\" width=\"1024\" height=\"213\"></p></p></li>\n<li><p><strong>Site won't load:</strong> If VoyantServer does not automatically launch your browser after running <code>VoyantServer.jar</code>, it could be one of two issues.</p>\n\n<ol>\n<li>The selected port is currently being used by another application or at capacity. Try exiting the application and opening it again. Changing the port may help fix this issue as well.</li>\n<li>If <em>Stop Server</em> is greyed out and the browser does not automatically load, exit the application, and try launching it again. This should fix the issue. You can also try to end the process from the Task Manager if the above approach fails.</li>\n</ol>\n</li>\n</ol>\n\n\n<p>Please let us know if you encounter other issues with VoyantServer. Send an email to <a href=\"mailto:voyanttools@gmail.com\">VoyantTools</a></p>\n","title":"VoyantServer"});