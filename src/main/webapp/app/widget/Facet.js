Ext.define('Voyant.widget.Facet', {
	extend: 'Ext.grid.Panel',
    mixins: ['Voyant.panel.Panel'],
    alias: 'widget.facet',
	statics: {
		i18n: {
		},
		api: {
			stopList: 'auto',
			query: undefined
		}
	},
	
	constructor: function(config) {
        this.callParent(arguments);
        Ext.applyIf(config || {}, {
        	includeTools: [], // don't show tools in header
        	rowLines: false,
			subtitle: undefined
        });
        this.mixins['Voyant.panel.Panel'].constructor.apply(this, [config]);
	},
	
	rowLines: false,
	
    initComponent: function(){

    	var me = this;
    	if (!this.store) {
    		this.store = new Ext.create("Voyant.data.store.CorpusFacets", {
    			proxy: {
    				extraParams: {
    					facet: this.facet
    				}
    			},
    			parentPanel: this
    		})
    		this.store.getProxy().on("exception", function(proxy, request, operation, eOpts) {
		    	me.showResponseError("Unable to fetch facet: "+me.facet, request);
    		})
    	}
    	
        Ext.applyIf(this, {
        	emptyText: this.localize("emptyText"),
        	hideHeaders: true,
        	selType: 'checkboxmodel',
        	columns: [
				{ renderer: function(value, metaData, record) {return "("+record.getInDocumentsCount()+") "+record.getLabel()}, flex: 1 }
        	]
        });
        this.callParent();
        
        if (this.corpus) {
        	this.setStoreCorpus(this.corpus);
        }
        
        this.on('loadedCorpus', function(src, corpus) {
        	this.setStoreCorpus(corpus);
        }, this);
        
        this.on("query", function(src, query) {
        	this.setApiParam("query", query);
        	// not getting set from beforeload, so set params here
        	this.store.load({
        		params: this.getApiParams()
        	})
        })
    },
    
    setStoreCorpus: function(corpus) {
//    	this.callParent(arguments)
    	if (this.getStore()) {
        	this.getStore().setCorpus(corpus);
        	this.getStore().load();
    	}
    }
});
