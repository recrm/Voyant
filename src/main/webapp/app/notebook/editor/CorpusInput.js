Ext.define('Voyant.notebook.editor.CorpusInput', {
	extend: 'Voyant.notebook.editor.CachedInput',
	alias: 'widget.notebookcorpusinput', 
	mixins: ['Voyant.util.Localization'],
	config: {
		corpusId: undefined,
		type: undefined,
		value: undefined
	},
	statics: {
		i18n: {
			corpusInput: 'Corpus Input',
			corpusText: 'Text',
			corpusVariable: 'Variable',
			corpusFile: 'File',
			corpusId: 'Corpus ID',
			notebookVariables: 'Notebook Variables'
		}
	},

	constructor: function(config) {
		Ext.apply(this, {
			layout: {
				type: 'hbox',
				align: 'begin',
				pack: 'start'
			},
			items: [{
				xtype: 'combo',
				itemId: 'type',
				fieldLabel: this.localize('corpusInput'),
				labelAlign: 'right',
				margin: '5px',
				queryMode: 'local',
				allowBlank: false,
				editable: false,
				forceSelection: true,
				value: 'text',
				store: [['text',this.localize('corpusText')],['variable',this.localize('corpusVariable')],['file',this.localize('corpusFile')],['id',this.localize('corpusId')]],
				listeners: {
					change: function(rg, val) {
						rg.nextSibling().getLayout().setActiveItem(val);
						this.setType(val);
						this.setCorpusId(undefined);
					},
					scope: this
				}
			},{
				xtype: 'container',
				layout: 'card',
				defaultType: 'container',
				itemId: 'cards',
				flex: 1,
				margin: '5px 5px 5px 0',
				items: [{
					itemId: 'text',
					items: [{
						xtype: 'textfield',
						width: '75%',
						name: 'text',
						fieldLabel: '',
						listeners: {
							change: function() {
								this.setCorpusId(undefined);
							},
							scope: this
						}
					}]
				},{
					itemId: 'variable',
					items: [{
						xtype: 'combo',
						name: 'variable',
						fieldLabel: '',
						triggerAction: 'all',
						queryMode: 'local',
						editable: false,
						forceSelection: true,
						emptyText: this.localize('notebookVariables'),
						store: { fields: ['text'] }
					}],
					listeners: {
						activate: function(crd) {
							this.populateVariables();
						},
						change: function() {
							this.setCorpusId(undefined);
						},
						scope: this
					}
				},{
					itemId: 'file',
					xtype: 'notebookfileinput',
					listeners: {
						change: function() {
							this.setCorpusId(undefined);
						},
						scope: this
					}
				},{
					itemId: 'id',
					items: [{
						xtype: 'textfield',
						width: '75%',
						name: 'id',
						fieldLabel: '',
						listeners: {
							change: function() {
								this.setCorpusId(undefined);
							},
							scope: this
						}
					}]
				}]
			}],
			listeners: {
				boxready: function(cmp) {
					if (config.type) {
						cmp.down('#type').setValue(config.type);
					}
					if (config.value) {
						var formField = cmp.down('#'+config.type).child();
						if (formField.setRawValue) {
							formField.setRawValue(config.value);
						} else {
							formField.setValue(config.value);
						}
					}
					// set corpusId last, because of change listeners
					if (config.corpusId) {
						cmp.setCorpusId(config.corpusId);
					}

					cmp.up('notebookdatawrapper').results.on('sandboxMessage', function(msg) {
						// use sandbox listener to capture corpusId
						if (msg.type === 'result') {
							var corpusVarName = cmp.up('notebookdatawrapper').getDataName();
							var corpusVal = msg.variables.find(function(variable) { return variable.name === corpusVarName });
							Spyral.Util.blobToString(corpusVal.value).then(function(strVal) {
								var corpusJson = JSON.parse(strVal);
								cmp.setCorpusId(corpusJson.corpusid);
							});
						}
					});

					cmp.up('notebook').on('notebookRun', function(notebook) {
						cmp.populateVariables();
					})
				}
			}
		});

		this.callParent(arguments);
	},

	populateVariables: function() {
		var cmp = this.up('notebookdatawrapper');
		var prev = cmp.previousSibling();
		var variables = [];
		if (prev !== null) {
			variables = cmp.up('notebook').getNotebookVariables(cmp).map(function(vr) { return [vr.name] });
		}
		var combo = this.down('#variable combo');
		combo.getStore().loadRawData(variables);
		
		var resetCombo = true;
		var currValue = combo.getRawValue();
		variables.forEach(function(variable) {
			if (variable.indexOf(currValue) !== -1) {
				resetCombo = false;
			}
		})
		if (resetCombo) {
			combo.setValue(undefined);
		}
	},

	getCode: function(varName) {
		var dfd = new Ext.Deferred();

		if (this.getCorpusId()) {
			var val = varName+'=new Spyral.Corpus("'+this.getCorpusId()+'")';
			dfd.resolve(val);
		} else {
			var activeItem = this.down('#cards').getLayout().getActiveItem();
			var type = activeItem.getItemId();
			this.setType(type);
			if (type === 'variable') {
				var inputVarName = activeItem.down('combo').getValue();
				this.setValue(inputVarName);
				if (inputVarName !== '') {
					var val = 'Spyral.Corpus.load({input:'+inputVarName+'}).then(function(){'+varName+'=arguments[0]})';
					dfd.resolve(val);
				} else {
					dfd.reject('No variable specified!');
				}
			} else if (type === 'file') {
				var file = activeItem.getFile();
				this.setValue(activeItem.getFileName());
				Spyral.Util.blobToDataUrl(file).then(function(dataUrl) {
					var val = 'Spyral.Corpus.load(Spyral.Util.dataUrlToBlob(`'+dataUrl+'`)).then(function(){'+varName+'=arguments[0]})';
					dfd.resolve(val);
				}, function() {
					dfd.reject();
				});
			} else if (type === 'text') {
				var text = activeItem.down('textfield').getValue().replace('"', '\"');
				this.setValue(text);
				if (text !== '') {
					var val = 'Spyral.Corpus.load({input:"'+text+'"}).then(function(){'+varName+'=arguments[0]})';
					dfd.resolve(val);
				} else {
					dfd.reject('No text entered!');
				}
			} else if (type === 'id') {
				var id = activeItem.down('textfield').getValue().replace('"', '\"');
				this.setValue(id);
				if (id !== '') {
					var val = 'Spyral.Corpus.load({corpus:"'+id+'"}).then(function(){'+varName+'=arguments[0]})';
					dfd.resolve(val);
				} else {
					dfd.reject('No ID entered!');
				}
			}
		}

		return dfd.promise;
	},

	getInput: function() {
		return {
			corpusId: this.getCorpusId(),
			type: this.getType(),
			value: this.getValue()
		}
	}
});