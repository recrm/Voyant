Ext.define('Voyant.panel.CorpusSet', {
	extend: 'Ext.panel.Panel',
    requires: ['Voyant.panel.VoyantTabPanel','Voyant.panel.Cirrus', 'Voyant.panel.Summary', 'Voyant.panel.CorpusTerms', 'Voyant.panel.Reader', 'Voyant.panel.Documents', 'Voyant.panel.Trends', 'Voyant.panel.Contexts', 'Voyant.panel.Phrases', 'Voyant.panel.DocumentTerms','Voyant.panel.CorpusCollocates','Voyant.panel.CollocatesGraph','Voyant.panel.StreamGraph','Voyant.panel.TermsBerry'],
	mixins: ['Voyant.panel.Panel'],
    alias: 'widget.corpusset',
	isConsumptive: true,
	statics: {
		i18n: {
		},
		api: {
			panels: undefined
		},
		glyph: 'xf17a@FontAwesome'
	},
	constructor: function(config) {
        this.callParent(arguments);
    	this.mixins['Voyant.panel.Panel'].constructor.apply(this, arguments);
	},
	layout: 'border',
	header: false,
	items: [{
    	region: 'west',
    	flex: 3,
    	layout: 'fit',
        moreTools: ['cirrus','corpusterms'],
        xtype: 'voyanttabpanel',
    	split: {width: 5},
    	tabBarHeaderPosition: 0,
    	items: [{
	    	xtype: 'cirrus'
    	},{
	    	xtype: 'corpusterms'
    	}, {
    		xtype: 'collocatesgraph'
    	}]
    },{
        region: 'center',
        flex: 3,
        layout: 'fit',
        xtype: 'voyanttabpanel',
    	tabBarHeaderPosition: 0,
        items: [{
	        xtype: 'reader' // termsradio added and set to default during loadedCorpus below when in non-consumptive mode
        },{
	        xtype: 'termsberry'
        }]
    }, {
    	region: 'east',
    	flex: 3,
    	layout: 'fit',
        xtype: 'voyanttabpanel',
    	split: {width: 5},
    	tabBarHeaderPosition: 0,
    	moreTools: ['trends','collocatesgraph'],
        items: [{
	    	xtype: 'trends'
        },{
	    	xtype: 'documentterms'
        }]
    }, {
    	region: 'south',
    	flex: 2,
    	split: {width: 5},
    	layout: 'border',
//    	layout: {
//    		type: 'hbox',
//    		align: 'stretch'
//    	},
    	items: [{
				layout: 'fit',
				region: 'center',
    			flex: 1,
    	        xtype: 'voyanttabpanel',
    	    	split: {width: 5},
    	    	tabBarHeaderPosition: 0,
    			moreTools: ['summary','documents','phrases'],
    			items: [{
	    			xtype: 'summary'
    			},{
	    			xtype: 'documents'
    			},{
	    			xtype: 'phrases'
    			}]
    		},{
				layout: 'fit',
				region: 'east',
    			flex: 1,
    	        xtype: 'voyanttabpanel',
    	    	split: {width: 5},
    	    	tabBarHeaderPosition: 0,
    			moreTools: ['contexts','documentterms','correlations'],
    			items: [{
	    			xtype: 'contexts'
    			},{
	    			xtype: 'bubblelines' // is set to default during loadedCorpus below when in non-consumptive mode
    			},{
	    			xtype: 'corpuscollocates'
    			}]
    	}]
    }],
    listeners: {
    	boxready: function() {
    		var panelsString = this.getApiParam("panels");
    		if (panelsString) {
    			var panels = panelsString.toLowerCase().split(",");
    			var tabpanels = this.query("voyanttabpanel");
    			for (var i=0, len=panels.length; i<len; i++) {
    				var panel = panels[i];
    				if (panel && Ext.ClassManager.getByAlias('widget.'+panel) && tabpanels[i]) {
    					var tabpanel = tabpanels[i];
    					if (tabpanel.getActiveTab().isXType(panel)) {continue;} // already selected
    					tabpanel.items.each(function(item, index) {
    						if (item.isXType(panel)) {
    							this.setActiveTab(index)
    							return false
    						}
    					}, tabpanel)
    					if (tabpanel.getActiveTab().isXType(panel)) {continue;} // already switched
    					tabpanel.getActiveTab().replacePanel(panel); // switch to specified panel
    				}
    			}
    		}
    		// add an easter egg
    		var cirrus = this.down('cirrus');
    		var me = this;
    		if (cirrus) {
				var imageBaseUrl = this.getApplication().getBaseUrl()+'resources/images/';
    			var toolbar = cirrus.down('toolbar');
    			toolbar.add({xtype: 'tbfill'})
    			toolbar.add({
    				text: ' ',
    				listeners: {
    					click: {
    						fn: function() {
	        					me.add({
	        						region: 'north',
	        						width: '100%',
	    							html: '<div align="center"><table><tr><td><img src="'+imageBaseUrl+'stefan.jpg" style="height: 60px"></td><td style="text-align: center; padding-left: 2em; padding-right: 2em;">By Athena, you found us hidden<br>up here between the panels!</td><td><img src="'+imageBaseUrl+'geoffrey.jpg" style="height: 60px"></td></tr></table></div>'
	        					})
	        				}, single: true
    					},
    					render: function(b) {
    						b.getTargetEl().dom.className=''
    					}
    				}
    			});
    		}
    		
    	},
    	loadedCorpus: function(src, corpus) {
    		if (this.hasCorpusAccess(corpus)==false && !this.getApiParam('panels')) {
    			var tabpanels = this.query("voyanttabpanel");
//    			tabpanels[1].add({xtype: 'termsradio'}); // reader
    			tabpanels[1].setActiveTab(1); // reader
    			tabpanels[1].getActiveTab().fireEvent("loadedCorpus", src, corpus); // make sure to load corpus
    			tabpanels[4].setActiveTab(1); // contexts
    		}
    		if (corpus.getDocumentsCount()>30) {
    			var bubblelines = this.down('bubblelines');
    			if (bubblelines) {
    				bubblelines.up('voyanttabpanel').remove(bubblelines)
    			}
    		}
    	},
    	panelChange: function(src) {
    		var panels = [];
    		this.query("voyanttabpanel").forEach(function(item) {
    			panels.push(item.getActiveTab().xtype)
    		})
    		this.getApplication().setApiParam('panels', panels.join(','))
    	}
    }
})