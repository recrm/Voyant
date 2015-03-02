Ext.define('Voyant.panel.CorpusCreator', {
	extend: 'Ext.form.Panel',
	requires: ['Ext.form.field.File'],
	requires: ['Voyant.data.model.Corpus'],
	mixins: ['Voyant.panel.Panel'],
	alias: 'widget.corpuscreator',
    statics: {
    	i18n: {
    		title: {en: "Add Texts"},
    		helpTip: {en: "This tool allows you to create a corpus in one of three ways:<ol><li>by typing or pasting text into the text box and clicking <i>Reveal</i>; if each line in the text box is a URL, text is fetched from those URLs, otherwise the contents are treated as a single document</li><li>click the <i>Open</i> button to open an existing corpus</li><li>click the <i>Upload</i> button to upload one or more files from you computer (you can select multiple files by using the Ctrl and/or Shift keys)</li></ul>"},
    		emptyInput: {en: "Type in one or more URLs on separate lines or paste in a full text."},
    		uploadingCorpus: {en: "Uploading corpus…"},
    		reveal: {en: "Reveal"}
    	}
    },
    config: {
    	
    },
    
    constructor: function(config) {
        this.callParent(arguments);
        config = config || {};
    	this.mixins['Voyant.panel.Panel'].constructor.call(this, Ext.apply(config, {includeTools: {gear: true, help: true}}));
    },
    
    initComponent: function() {
        var me = this;

        Ext.apply(me, {
    		title: this.localize('title'),
    		width: 800,
    		frame: true,
    		border: true,
    		frameHeader: true,
    		layout: {
    			type: 'vbox',
    			align: 'middle'
    		},
	    	dockedItems: [{
	    		xtype: 'toolbar',
                dock: 'bottom',
    	    	buttonAlign: 'right',
    	    	defaultButtonUI : 'default',
	    		items: [{
	    			text: 'Open',
                    glyph: 'xf115@FontAwesome', // not visible
	    			tooltip: 'Select an exsting corpus',
	    			handler: function() {
	    				Ext.create('Ext.window.Window', {
	    				    title: 'Open an Existing Corpus',
	    				    layout: 'fit',
	    				    modal: true,
	    				    items: {  // Let's put an empty grid in just to illustrate fit layout
	    				        xtype: 'form',
	    				        margin: '5,5,5,5',
	    				        items: {
	    				            xtype:'combo',
	    				            labelWidth: 150,
	    				            fieldLabel:'Choose a corpus:',
	    				            name:'corpus',
	    				            queryMode:'local',
	    				            store:[['shakespeare',"Shakespeare's Plays"]],
	    				            
	    				            forceSelection:true
	    				        },
	    				        buttons: [
	    				        	{
	    				        		text: 'Open',
	    			                    glyph: 'xf00c@FontAwesome',
	    				        		handler: function(btn) {
	    				        			var form = btn.up('form').getForm();
	    				        			var corpus = btn.up('form').getForm().getValues().corpus
	    				        			if (corpus!='') {
	    				        				me.loadCorpus({corpus: corpus})
		    				        			btn.up('window').close();
	    				        			}
	    				        			else {
	    				    	        		Ext.Msg.show({
	    				    	        		    title:'Select a Corpus',
	    				    	        		    message: 'Please be sure to select a corpus.',
	    				    	        		    buttons: Ext.Msg.OK,
	    				    	        		    icon: Ext.Msg.ERROR
	    				    	        		});
	    				        			}
	    				        		},
	    				        		flex: 1
	    				            },{
	    				        		text: 'Cancel',
	    			                    glyph: 'xf00d@FontAwesome',
	    				        		flex: 1,
	    				        		handler: function(btn) {
	    				        			btn.up('window').close();
	    				        		}
	    				        	}
	    				        ]
	    				    }
	    				}).show();
	    			}
	    		},{
    	        	xtype: 'filefield',
                    glyph: 'xf093@FontAwesome',
    	        	name: 'upload',
        	    	buttonOnly: true,
        	    	hideLabel: true,
        	    	buttonText: 'Upload',
        	    	tooltip: 'test',
        	    	listeners: {
        	    		render: function(filefield) {
        	    			filefield.fileInputEl.dom.setAttribute('multiple', true);
        	    		      Ext.QuickTips.register({
        	      		        target: filefield.getEl(),
        	      		        text: 'Upload one or more documents from your computer',
        	      		        enabled: true,
        	      		        showDelay: 20,
        	      		        trackMouse: true,
        	      		        autoShow: true
        	      		      });
        	            },
        	            change: function(filefield, value) {
        	            	if (value) {
            	            	var form = filefield.up('form').getForm();
            	            	var view = me.getApplication().getViewport();
            	            	if (form.isValid()) {
            	            		view.mask(me.localize('uploadingCorpus'))
            	            		form.submit({
            	            			url: me.getTromboneUrl(),
            	            			params: {tool: 'corpus.CorpusCreator'},
            	            			failure: function(form, action) { // we always fail because of content-type
                        	            	view.unmask()
            	            				if (action.result) {
            	            					me.loadCorpus({corpus: action.result.stepEnabledCorpusCreator.storedId})
            	            				}
            	            			}
            	            		})
            	            	}
        	            	}
        	            }
        	    	}
	    		},'->',{
	    	    	xtype: 'button',
	    	    	scale: 'large',
                    glyph: 'xf00c@FontAwesome',
	    	    	text: this.localize('reveal'),
	    	    	handler: function(btn) {
	    	        	var input = btn.up('form').down('#input').getValue();
	    	        	if (input !== '') {
	    	        		me.loadCorpus({input: input});
	    	        	}
	    	        	else {
	    	        		Ext.Msg.show({
	    	        		    title:'No Text Provided',
	    	        		    message: 'Please provide text in the text box (or choose open or upload).',
	    	        		    buttons: Ext.Msg.OK,
	    	        		    icon: Ext.Msg.ERROR
	    	        		});
	    	        	}
	    	    	}
	    	    }]
	    	}],
	    	items: {
    	    	xtype: 'textareafield',
    	    	width: 800,
    	    	height: 100,
    	    	itemId: 'input',
    	    	emptyText: this.localize('emptyInput')
	    	}    
        });
        
        me.callParent(arguments);
    },
    
    loadCorpus: function(params) {
		var app = this.getApplication();
    	var view = app.getViewport();
		view.mask();
		new Corpus(params).then(function(corpus) {
			view.unmask();
			app.dispatchEvent('loadedCorpus', app, corpus);
		}).fail(function(message, response) {
			view.unmask();
			app.showErrorResponse({message: message}, response);
		});
    }
    
});