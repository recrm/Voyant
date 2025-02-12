Ext.define('Voyant.panel.Veliza', {
	extend: 'Ext.panel.Panel',
	mixins: ['Voyant.panel.Panel'],
    xtype: 'veliza',
	autoScroll: true,
    statics: {
    	i18n: {
    	},
    	api: {
    		script: '',
    		message: undefined
    	},
		glyph: 'xf0e6@FontAwesome'
    },
    config: {
    	previous: []
    },
    
    constructor: function() {
        this.callParent(arguments);
    	this.mixins['Voyant.panel.Panel'].constructor.apply(this, arguments);
    	
        var me = this;
        Ext.apply(this, {
    		title: this.localize('title'),
    		glyph: 'xf0e6@FontAwesome',
    	    layout: {
    	        type: 'border',
    	        align: 'stretch'
    	    },
    		items: [{
    			itemId: 'chat',
	    		html: "<form class='chat'></form>",
	    		region: 'center',
	    		flex: 2,
	    		autoScroll: true
    		},{
    			itemId: 'script',
    			xtype: 'form',
    			region: 'east',
    			split: true,
    			title: this.localize('scriptEditor'),
        	    layout: {
        	        type: 'vbox',
        	        align: 'stretch'
        	    },
    			items: [{
    				html: new Ext.XTemplate(this.localize('scriptIntro')).apply([me.getBaseUrl()+"docs/tutorial-veliza.html"])
    			},{
    				xtype: 'textarea',
    				name: 'editor',
    				fieldStyle: "white-space: pre",
    				value: me.getApiParam('script'),
    				flex: 1,
    				listeners: {
    					afterrender: function(editor) {
    						var corpus = me.getApiParam('corpus');
    						if (!corpus) {return}
    						editor.mask(me.localize('loadingScript'));
    						Ext.Ajax.request({
    							url: me.getTromboneUrl(),
    							params: {
    								tool: 'corpus.Veliza',
    								script: me.getApiParam('script'),
    								corpus: corpus
    							}
    						}).then(function(response) {
    							var obj = Ext.decode(response.responseText);
    							if (obj && obj.veliza && obj.veliza.script) {
    								editor.setValue(obj.veliza.script);
    								editor.resetOriginalValue();
    							} else if (obj && obj.veliza && obj.veliza.id) {
    								me.setApiParam('script', obj.veliza.id);
    							} else {
    								me.showError(me.localize('unableFetchScript'));
    							}
    							editor.unmask()
    						}, function(response) {
    							me.showError(response)
    						})
    					},
    					scope: this
    				}
    			}],
    			collapsed: true,
    			collapsible: true,
    			flex: 1
    		}],
    		dockedItems: [{
                dock: 'bottom',
                xtype: 'toolbar',
                overflowHandler: 'scroller',
                items: [{
        			xtype: 'textfield',
        			itemId: 'chatfield',
        			emptyText: this.localize("typeAndEnter"),
        			flex: 1,
        			listeners: {
                        specialkey: function(field, e){
                            if (e.getKey() == e.ENTER) {
                            	me.handleUserSentence(field.getValue())
                            	field.setValue("");
                            }
                        }
                    }
                },{
        			xtype: 'button',
        			text: this.localize('send'),
        			handler: function() {
        				var tf = this.up("toolbar").down('textfield');
        				me.handleUserSentence(tf.getValue(), false);
        				tf.setValue('');
        			}
        		},{
        			xtype: 'button',
        			text: this.localize('fromCorpus'),
        			handler: function() {
        				me.handleUserSentence("", true)
        			}
        		}]
    		}]
        })
             
        this.callParent();
        
    	this.on('boxready', function(cmp) {
    		cmp.addSentence("fromThem", "Hello, I'm Veliza, and I'm here to talk to you about your texts (you may know my sister <a href='https://en.wikipedia.org/wiki/ELIZA' target='_blank'>Eliza</a> she's a famous psychotherapist). I'm just learning to talk about text documents, but please, let me know about any anxieties you're feeling about your texts. Type a message in the box below and hit enter. Or, if you're feeling playful, hit the <i>from text</i> bottom in the lower right-hand corner to fetch a sentence from the corpus.");
    		this.sendApiParamMessage();
    	})

    }, 
    
    sendApiParamMessage: function() {
		if (this.getApiParam('message')) {
			if (this.getCorpus()) {
				var sentences = Ext.Array.from(this.getApiParam('message'));
				var sentence = sentences.shift();
				if (sentence) {
					if (sentences) {
						this.setApiParam("message", sentences);
					}
					this.handleUserSentence(sentence, undefined, true)
					
				}
			} else {
				// try to wait for the corpus to be loaded
				Ext.defer(this.sendApiParamMessage, 100, this, [true])
			}
		}
    },
    
    
    handleUserSentence: function(sentence, fromCorpus, noScroll) {
    	sentence = sentence.trim();
    	if (sentence || fromCorpus) {
    		if (sentence) {
    	    	this.addSentence("myMessage", sentence);
    		}
	    	this.mask();
    		var me = this;
    		var editor = this.getComponent('script').down('textarea');
    		Ext.Ajax.request({
    			url: this.getApplication().getTromboneUrl(),
    			params: {
    				corpus: me.getCorpus() ? me.getCorpus().getId() : undefined,
    				tool: 'corpus.Veliza',
    				sentence: sentence,
    				//previous: this.getPrevious(),
    				fromCorpus: fromCorpus ? true : false,
    				script: editor.isDirty() ? editor.getValue() : this.getApiParam('script'),
    				noCache: Ext.id()
    			},
    		    success: function(response, opts) {
    		    	me.unmask();
    		    	var response = Ext.decode(response.responseText);
    		    	if (response.veliza.id) {
    		    		me.setApiParam('script', response.veliza.id);
    		    		editor.resetOriginalValue();
    		    	}
    		    	var veliza = response.veliza.response;
    		    	var hidden = veliza.match(/<\!-- (.+?) -->/);
    		    	if (hidden) {
    		    		var json =  Ext.decode(hidden[1]);
    		    		for (key in json.params) {
    		    			json.params[key] = json.params[key].trim();
    		    		}
    		    		veliza += "<br/><iframe width='"+(json.width ? json.width : '100%')+"' height='"+(json.height ? json.height : '250px')+"' src='"+me.getApplication().getBaseUrl()+'tool/'+json.tool+'/?corpus='+me.getCorpus().getId()+'&minimal=true&'+Ext.Object.toQueryString(json.params)+"'></iframe>"
    		    	}
    		    	var sentence = response.veliza.sentence;
    		    	me.setPrevious(response.veliza.previous);

    		    	if (fromCorpus) {
    		    		meta = response.veliza.docIndex > -1 ? me.getCorpus().getDocument(response.veliza.docIndex).getShortLabel() : undefined;
    		    		me.addSentence("myMessage", sentence, meta);
        			    	Ext.Function.defer(function() {
                		    	this.addSentence("fromThem", veliza);
                		    	if (!noScroll) {
                    		    	this.body.scroll('b', Infinity)                		    		
                		    	}
        			    	}, 500, me);
    		    	} else {
        		    	me.addSentence("fromThem", veliza);
        		    	if (!noScroll) {
            		    	me.body.scroll('b', Infinity)                		    		
        		    	}
    		    	}
    		    	if (me.getApiParam("message")) { // any more messages to show?
    		    		me.sendApiParamMessage();
    		    	}
    		    },
    		    failure: function(response, opts) {
    		    	me.showResponseError("Unable to get response from Veliza.", response);
    		    }
    		})
    	}
    },

    addSentence: function(speaker, sentence, meta) {
    	var body = this.getComponent('chat').body;
    	var el = body.down("form").insertHtml('beforeEnd', '<div class="message"><div class="'+speaker+'"><p>'+sentence+'</p>'+(meta ? "<date>"+meta+"</date>" : "")+'</div></div>', true);
    	body.scroll('b', Infinity);
    }
});