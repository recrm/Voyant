Ext.define('Voyant.panel.DreamScape', {
    extend: 'Ext.Panel',
    xtype: 'dreamscape',
    mixins: ['Voyant.panel.Panel'],
    statics: {
        i18n: {
        },
        api: {
            stopList: 'auto',
            hide: [],
            author: undefined,
            title: undefined,
            keyword: undefined,
            pubDate: undefined,
            citiesMaxCount: 500,
            minPopulation: 10000,
            citiesMinFreq: 1,
            connectionsMaxCount: 2500,
            connectionsMinFreq: 1,
            millisPerAnimation: 2000,
            annotationsId: undefined,
            source: undefined,
            overridesId: undefined,
            filterHasLowerCaseForm: true,
            filterIsPersonName: true,
            preferredCoordinates: undefined
        },
        glyph: 'xf124@FontAwesome'
    },
    config: {
        map: undefined, // OL object, should be set during init
        overlay: undefined, // OL object, should be set during init
        contentEl: undefined, // EXTJS Element, where info is shown
        isOpenLayersLoaded: false,
        isArcLoaded: false,
        animationDelay: 25, // determines the delay between animation calls
        isProj4Loaded: false,
        projection: undefined,
        filterWidgets: new Ext.util.MixedCollection(),
        drawInteraction: undefined,
        drawMode: false,
        //currentAnnotation: undefined,
        baseLayers: {},
        annotations: undefined, // set during init if any values are stored, shouldn't be looked up (getAnnotations) after init
        annotationsLoadedIfAvailable: false, // check api and local storage,
        overrides: {}
    },

    html: '<div class="map"></div><div class="ticker"></div>',

    cls: 'dreamscape',

    constructor: function(config) {
        this.callParent(arguments);
        this.mixins['Voyant.panel.Panel'].constructor.apply(this, arguments);
        Ext.Loader.loadScript({
            url: this.getBaseUrl()+"resources/openlayers/ol.js",
            onLoad: function() {
                // Utility function to get layer by id
                if (ol.Map.prototype.getLayer === undefined) {
                    ol.Map.prototype.getLayer = function (id) {
                        var layer = undefined;
                        this.getLayers().forEach(function(lyr) {
                            if (id === lyr.get('id')) {
                                layer = lyr;
                            }
                        });
                        return layer;
                    }
                }
                this.setIsOpenLayersLoaded(true);
            },
            scope: this
        });
        Ext.Loader.loadScript({
            url: this.getBaseUrl()+"resources/dreamscape/arc.js",
            onLoad: function() {
                this.setIsArcLoaded(true);
            },
            scope: this
        });
        Ext.Loader.loadScript({
            url: "https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.4.4/proj4.js",
            onLoad: function() {
                this.setIsProj4Loaded(true);
            },
            scope: this
        });
        var annotationsId = this.getApiParam("annotationsId");
        if (annotationsId) {
            var me = this;
            Ext.Ajax.request({
                url: this.getTromboneUrl(),
                params: {
                    tool: 'resource.StoredResource',
                    retrieveResourceId: annotationsId
                },
                scope: this
            }).then(function(response) {
                var data = Ext.JSON.decode(response.responseText);
                if (data.storedResource.resource) {
                    var annotations = Ext.JSON.decode(data.storedResource.resource)
                    if (Ext.isString(annotations)) {
                        annotations = Ext.JSON.decode(annotations);
                        me.setAnnotations(annotations)
                    }
                }
                if (!me.getAnnotations()) {
                    me.showError(me.localize("annotationsLoadFailed"));
                }
                me.setAnnotationsLoadedIfAvailable(true);
            }, function(response) {
                me.showResponseError(me.localize("annotationsLoadFailed"), response);
                me.setAnnotationsLoadedIfAvailable(true);
            });
        } else {
            this.setAnnotationsLoadedIfAvailable(true);
        }
    },
    initComponent: function() {
        var me = this;
        this.mixins['Voyant.util.Api'].constructor.apply(this, arguments); // we need api
        var hide = this.getApiParam("hide");
        if (Ext.isString(hide)) {
            hide = hide.split(',');
            this.setApiParam('hide');
        }
        var iconBase = this.getApplication().getBaseUrl()+"resources/dreamscape/";
        Ext.apply(this, {
            title: this.localize('title'),
            bodyBorder: true,
            dockedItems: [{
                dock: 'bottom',
                xtype: 'toolbar',
                itemId: 'bottomToolbar',
                overflowHandler: 'scroller',
                items: [
                    {
                        text: this.localize('display'),
                        tooltip: this.localize('displayTip'),
                        glyph: 'xf013@FontAwesome',
                        menu: {
                            defaults: { // these are intended for the radiogroups after the separator (and should be ignored by the menu items above the separator)
                                xtype: 'menucheckitem',
                                checkHandler: function(item, checked) {
                                    var map = this.getMap(), id = item.getItemId();
                                    this.getFilterWidgets().each(function(filter) {
                                        map.getLayer(filter.getId()+"-"+id).setVisible(checked);
                                    })
                                    var hide = Ext.Array.from(this.getApiParam('hide'));
                                    if (checked) {
                                        hide = Ext.Array.remove(hide, id);
                                    } else {
                                        hide.push(id);
                                        hide = Ext.Array.unique(hide);
                                    }
                                    this.setApiParam("hide", hide);
                                    item.getMenu().setDisabled(checked==false);
                                },
                                scope: this,
                                checked: true
                            },
                            items: [{
                                xtype: 'menuitem',
                                text: this.localize('baseLayer'),
                                tooltip: this.localize('baseLayerTip'),
                                glyph: 'xf279@FontAwesome',
                                menu: {
                                    items: [{
                                        xtype: 'radiogroup',
                                        columns: 1,
                                        vertical: true,
                                        defaults: {
                                            cls: 'map-menu-item',
                                            handler: function(item, checked) {
                                                if(checked){
                                                    var panel = this;
                                                    id = item.getItemId();
                                                    var layer = this.getBaseLayers()[id];
                                                    if (layer) {
                                                        layer.setOpacity(1);
                                                        this.getMap().getLayers().setAt(0, layer);
                                                    }
                                                    if(id == 'osm' || id == 'arcGIS') {
                                                        this.getMap().getLayer("overlayLayer").setVisible(false);
                                                    } else {
                                                        this.getMap().getLayer("overlayLayer").setVisible(true);
                                                    }
                                                }
                                            },
                                            listeners: {
                                                afterrender: function(item) {
                                                    Ext.create('Ext.tip.ToolTip', {
                                                        target: item,
                                                        html: '<div class="map-menu-item-tooltip '+item.getItemId()+'"></div>',
                                                        anchor: 'right'
                                                    });
                                                }
                                            },
                                            scope: this
                                        },
                                        items: [
                                            // {
                                            //     boxLabel: this.localize('watercolor'),
                                            //     itemId: 'watercolor',
                                            //     cls: ['map-menu-item','watercolor'],
                                            //     checked: true
                                            // },
                                            {
                                                boxLabel: this.localize('wms4326'),
                                                cls: ['map-menu-item','wms4326'],
                                                itemId: 'wms4326'
                                            },{
                                                boxLabel: this.localize('osm'),
                                                cls: ['map-menu-item','osm'],
                                                itemId: 'osm',
                                                checked: true
                                            },{
                                                boxLabel: this.localize('arcGIS'),
                                                cls: ['map-menu-item','arcGIS'],
                                                itemId: 'arcGIS'
                                            }
                                        ]
                                    }]
                                }
                            },{
                                xtype: 'menuitem',
                                text: this.localize('projection'),
                                tooltip: this.localize('projectionTip'),
                                glyph: 'xf0ac@FontAwesome',
                                menu: {
                                    items: [{
                                        xtype: 'radiogroup',
                                        columns: 1,
                                        vertical: true,
                                        defaults: {
                                            handler: function(item, checked) {
                                                if(checked){
                                                    var panel = this;
                                                    id = item.getItemId();
                                                    var view = undefined
                                                    if (id === "webMercatorProjection") {
                                                        this.setProjection(ol.proj.get('EPSG:3857'));
                                                    } else if (id === "mercatorProjection") {
                                                        this.setProjection(ol.proj.get('EPSG:4326'));
                                                    } else if (id === "gallPetersProjection") {
                                                        proj4.defs('cea',"+proj=cea +lon_0=0 +lat_ts=45 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs");
                                                        var gallPetersProjection = ol.proj.get('cea');
                                                        this.setProjection(gallPetersProjection);
                                                    } else if (id === "sphereMollweideProjection") {
                                                        proj4.defs('ESRI:54009', '+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs');
                                                        var sphereMollweideProjection = ol.proj.get('ESRI:54009');
                                                        sphereMollweideProjection.setExtent([-18e6, -9e6, 18e6, 9e6]);
                                                        this.setProjection(sphereMollweideProjection);
                                                    }
                                                    var newProjExtent = this.getProjection().getExtent();
                                                    // If using Mercator projection, earth with is measured in degrees, otherwise it's in meters
                                                    var earthWidth = this.getProjection() == ol.proj.get("EPSG:4326") ? 360 : 40075016.68557849;
                                                    var newView = new ol.View({
                                                        projection: this.getProjection(),
                                                        center: ol.extent.getCenter(newProjExtent || [0, 0, 0, 0]),
                                                        maxResolution: earthWidth / panel.body.dom.offsetWidth,
                                                        zoom: 0
                                                    });
                                                    var map = this.getMap();
                                                    var layers = map.getLayers();
                                                    layers.forEach(function(layer) {
                                                        if(layer.getSource().getFeatures){
                                                            var features = layer.getSource().getFeatures();
                                                            features.forEach(function(feature){
                                                                var newGeometry = feature.getGeometry().transform(map.getView().getProjection(), panel.getProjection());
                                                                feature.setGeometry(newGeometry);
                                                            })
                                                        }
                                                    });

                                                    map.setView(newView);
                                                }
                                            },
                                            listeners: {
                                                afterrender: function(item) {
                                                    Ext.create('Ext.tip.ToolTip', {
                                                        target: item,
                                                        html: '<div class="map-menu-item-tooltip '+item.getItemId()+'"></div>',
                                                        anchor: 'right'
                                                    });
                                                }
                                            },
                                            scope: this
                                        },
                                        items: [
                                            {
                                                boxLabel: this.localize('webMercatorProjection'),
                                                itemId: 'webMercatorProjection',
                                                cls: ['map-menu-item','webMercatorProjection'],
                                                checked: true
                                            },{
                                                boxLabel: this.localize('mercatorProjection'),
                                                cls: ['map-menu-item','mercatorProjection'],
                                                itemId: 'mercatorProjection'
                                            },{
                                                boxLabel: this.localize('gallPetersProjection'),
                                                cls: ['map-menu-item','gallPetersProjection'],
                                                itemId: 'gallPetersProjection'
                                            },{
                                                boxLabel: this.localize('sphereMollweideProjection'),
                                                cls: ['map-menu-item','sphereMollweideProjection'],
                                                itemId: 'sphereMollweideProjection'
                                            }
                                        ]
                                    }]

                                }
                            },'-',{
                                text: this.localize('cities'),
                                checked: Ext.Array.contains(hide, "cities")==false,
                                itemId: 'cities',
                                menu: {
                                    defaults: {
                                        labelAlign: 'right',
                                        labelWidth: 140
                                    },
                                    items: [{
                                        xtype: 'numberfield',
                                        fieldLabel: this.localize('citiesMaxCount'),
                                        minValue: 1,
                                        value: parseInt(this.getApiParam('citiesMaxCount')),
                                        listeners: {
                                            change: function(cmp, newVal, oldVal) {
                                                this.getFilterWidgets().each(function(filter) {
                                                    this.setApiParam('citiesMaxCount', newVal);
                                                    var currentCitiesCount = filter.getGeonames().getCitiesCount();
                                                    if (newVal<oldVal || newVal<=currentCitiesCount) {
                                                        this.filterUpdate(filter);
                                                    } else if (filter.getGeonames().hasMoreCities()) {
                                                        filter.loadGeonames()
                                                    } else {
                                                        cmp.setValue(currentCitiesCount); // will call change
                                                        this.toastInfo(new Ext.XTemplate(this.localize("allNCitiesShown")).apply([currentCitiesCount]))
                                                    }
                                                }, this);
                                            },
                                            scope: this
                                        }
                                    },{
                                        xtype: 'numberfield',
                                        fieldLabel: this.localize('citiesMinPopulation'),
                                        minValue: 1,
                                        value: parseInt(this.getApiParam('minPopulation')),
                                        listeners: {
                                            change: function(cmp, newVal, oldVal) {
                                                this.setApiParam('minPopulation', newVal);
                                                this.getFilterWidgets().each(function(filter) {
                                                    if (newVal>oldVal) {
                                                        this.filterUpdate(filter);
                                                    } else {
                                                        filter.loadGeonames()
                                                    }
                                                }, this);
                                            },
                                            scope: this
                                        }
                                    },{
                                        xtype: 'numberfield',
                                        fieldLabel: this.localize('citiesMinFreq'),
                                        minValue: 1,
                                        value: parseInt(this.getApiParam('citiesMinFreq')),
                                        listeners: {
                                            change: function(cmp, newVal, oldVal) {
                                                this.getFilterWidgets().each(function(filter) {
                                                    this.setApiParam('citiesMinFreq', newVal);
                                                    if (newVal>oldVal) {
                                                        this.filterUpdate(filter);
                                                    } else {
                                                        filter.loadGeonames()
                                                    }
                                                }, this);
                                            },
                                            scope: this
                                        }
                                    }]
                                }
                            },{
                                text: this.localize('connections'),
                                checked: Ext.Array.contains(hide, "connections")==false,
                                itemId: 'connections',
                                menu: {
                                    defaults: {
                                        labelWidth: 140,
                                        labelAlign: 'right'
                                    },
                                    items: [{
                                        xtype: 'numberfield',
                                        fieldLabel: this.localize('connectionsMaxCount'),
                                        minValue: 1,
                                        value: parseInt(this.getApiParam('connectionsMaxCount')),
                                        listeners: {
                                            change: function(cmp, newVal, oldVal) {
                                                this.getFilterWidgets().each(function(filter) {
                                                    this.setApiParam('connectionsMaxCount', newVal);
                                                    if (newVal>oldVal) {
                                                        this.filterUpdate(filter);
                                                    } else {
                                                        filter.loadGeonames()
                                                    }
                                                }, this);
                                            },
                                            scope: this
                                        }
                                    },{
                                        xtype: 'numberfield',
                                        fieldLabel: this.localize('connectionsMinFreq'),
                                        minValue: 1,
                                        value: parseInt(this.getApiParam('connectionsMinFreq')),
                                        listeners: {
                                            change: function(cmp, newVal, oldVal) {
                                                this.getFilterWidgets().each(function(filter) {
                                                    this.setApiParam('connectionsMinFreq', newVal);
                                                    if (newVal>oldVal) {
                                                        this.filterUpdate(filter);
                                                    } else {
                                                        filter.loadGeonames()
                                                    }
                                                }, this);
                                            },
                                            scope: this
                                        }
                                    }]
                                }

                            },{
                                xtype: 'menucheckitem',
                                checked: Ext.Array.contains(hide, "animation")==false,
                                text: this.localize('animations'),
                                itemId: 'animation',
                                menu: {
                                    defaults: {
                                        labelWidth: 170,
                                        labelAlign: 'right',
                                        width: 280
                                    },
                                    items: [{
                                        xtype: 'sliderfield',
                                        fieldLabel: this.localize('millisPerAnimation'),
                                        value: parseInt(this.getApiParam('millisPerAnimation')),
                                        minValue: this.getAnimationDelay(),
                                        maxValue: 10000,
                                        listeners: {
                                            change: function(cmp, newVal, oldVal) {
                                                this.getFilterWidgets().each(function(filter) {
                                                    filter.setMillisPerAnimation(newVal);
                                                    filter.animate();
                                                }, this);
                                            },
                                            scope: this
                                        }
                                    }]
                                }
                            }]
                        }
                    }, '-', {
                        text: this.localize('add'),
                        itemId: 'addFilter',
                        glyph: 'xf067@FontAwesome',
                        handler: function(cmp) {
                            var colors = this.getApplication().getColorPalette(this.getApiParam('palette'), true),
                                color = colors[0]; // default to first

                            // find an available color
                            for (var i=0, len=colors.length; i<len; i++) {
                                var hasMatch = false;
                                this.getFilterWidgets().each(function(filter) {
                                    if (filter.getColor()==colors[i]) {
                                        hasMatch = true;
                                        return false;
                                    }
                                })
                                if (hasMatch==false) {
                                    color = colors[i];
                                    break;
                                }
                            }
                            var filter = cmp.ownerCt.add({
                                xtype: 'geonamesfilter',
                                corpus: this.getCorpus(),
                                color: color,
                                listeners: {
                                    removeFilterWidget: function(filter) {
                                        var map = this.getMap(), id = filter.getItemId(), filters = this.getFilterWidgets();
                                        ["connections","cities","animation"].forEach(function(layer) {
                                            map.removeLayer(map.getLayer(id+"-"+layer));
                                        })
                                        filters.remove(filter);
                                        if (filters.getCount()==0) { // we always want at least one filter
                                            // simulate adding a filter
                                            this.getDockedComponent('bottomToolbar').getComponent('addFilter').click();
                                        }
                                    },
                                    filterUpdate: this.filterUpdate,
                                    scope: this
                                }
                            })
                            this.getFilterWidgets().add(filter.getId(), filter);

                            var map = this.getMap();

                            var connectionsLayer = new ol.layer.Vector({
                                source: new ol.source.Vector({
                                    wrapX: false,
                                    useSpatialIndex: true // needed to call getExtent()
                                }),
                                id: filter.getId()+"-connections",
                                visible: true,
                                opacity: 0.2,
                                style: this.travelStyleFunction.bind(this),
                                updateWhileAnimating: true, // optional, for instant visual feedback
                                updateWhileInteracting: true // optional, for instant visual feedback
                            });
                            map.addLayer(connectionsLayer);

                            var citiesLayer = new ol.layer.Vector({
                                source: new ol.source.Vector({
                                    wrapX: false,
                                    useSpatialIndex: true // needed to call getExtent()
                                }),
                                id: filter.getId()+"-cities",
                                opacity: 0.5,
                                style: this.cityStyleFunction.bind(this)
                            });
                            map.addLayer(citiesLayer);

                            var animationLayer = new ol.layer.Vector({
                                source: new ol.source.Vector({
                                    wrapX: false,
                                    useSpatialIndex: true // needed to call getExtent()
                                }),
                                id: filter.getId()+"-animation",
                                style:  this.animationStyleFunction.bind(this)
                            });
                            map.addLayer(animationLayer);
                        },
                        scope: this
                    }]
            }]
        });

        this.on("loadedCorpus", function(src, corpus) {
            this.tryInit();
        }, this);

        this.callParent();
    },

    tryInit: function() {
        if (this.getIsOpenLayersLoaded() && this.getIsArcLoaded() && this.getIsProj4Loaded() && this.getAnnotationsLoadedIfAvailable()) {
            var el = this.body.down('.map').setId(Ext.id());
            /*
            var overlayEl = this.body.down('.ol-popup');
            var overlay = new ol.Overlay({
                element: overlayEl.dom,
                autoPan: true,
                autoPanAnimation: {
                    duration: 250
                }
            });
            this.setOverlay(overlay);
            */
            var panel = this;
            // Add a click handler to hide the popup
            /*
            var closer = overlayEl.down(".ol-popup-closer").dom;
            closer.onclick = function() {
                overlay.setPosition(undefined);
                closer.blur();
                panel.getMap().getLayer("preview").getSource().clear();

//                if(panel.getDrawMode()) {
//                    panel.getDockedComponent('bottomToolbar').getComponent('annotate').click();
//                }
                return false;
            };
            */

            var baseLayers = this.getBaseLayers();
            baseLayers['osm'] = new ol.layer.Tile({
                preload: Infinity,
                source: new ol.source.OSM({
                    projection: "EPSG:3857"
                })
            });

            baseLayers['wms4326'] = new ol.layer.Tile({
                preload: Infinity,
                source: new ol.source.TileWMS({
                    url: 'https://ahocevar.com/geoserver/wms',
                    crossOrigin: '',
                    params: {
                        'LAYERS': 'ne:NE1_HR_LC_SR_W_DR',
                        'TILED': true
                    },
                    projection: 'EPSG:4326'
                })
            });

            // baseLayers['watercolor'] =  new ol.layer.Tile({
            //     preload: Infinity,
            //     source: new ol.source.Stamen({
            //         //cacheSize: 2048,
            //         layer: 'watercolor',
            //         projection: "EPSG:3857"
            //     })
            // });

            baseLayers['arcGIS'] = new ol.layer.Tile({
                preload: Infinity,
                source: new ol.source.TileArcGISRest({
                    url: "https://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer",
                    projection: "EPSG:3857"
                })
            });

//            this.setContentEl(overlayEl.down('.popup-content'));
            var map = new ol.Map({
                layers: [
                    baseLayers['osm'],

                    // new ol.layer.Tile({
                    //     preload: Infinity,
                    //     source: new ol.source.Stamen({
                    //         cacheize: 2048,
                    //         layer: 'toner-hybrid',
                    //         projection: "EPSG:3857"
                    //     }),
                    //     id: 'overlayLayer'
                    // })
                ],
//                overlays: [overlay],
                target: el.getId(),
                loadTilesWhileInteracting: true,
                view: new ol.View({
                    center: [0, 0],
                    maxResolution: 40075016.68557849 / panel.body.dom.offsetWidth,
                    zoom: 0
                })
            });

            map.on('singleclick', this.handleSingleClick, this);
            // Add a click handler to the map to render the popup
            map.on('singleclicking', function(event) {
                var pixel = event.pixel;
                var features = map.getFeaturesAtPixel(pixel);
                if (features) {
                		for (var i=0; i<features.length; i++) {
                			var feature = features[i];
                			if (feature.get("type")=='city') {
                				this.handleLocationClick(feature);
                			}
                		}
                		var feature = features[0];
                		debugger
                }
                
                if(features) {
                    var foundFeature = false;
                    features.forEach( function(feature) {
                        if( feature.get("type") === "city" && feature.get("selected")) { // city
                            panel.dispatchEvent("termsClicked", panel, [feature.get("forms").join("|")])
                            foundFeature = true;
                        } else if (feature.get("type") === "connection" && feature.get("selected") && !foundFeature) { // connection
                            panel.getFilterWidgets().each(function (filter) {
                                var geonames = filter.getGeonames();
                                if (geonames != null) {
                                    var occurrences = geonames.getAllConnectionOccurrences(feature.get("source"), feature.get("target"));
                                    var infos = "<ul>";
                                    var docIndex = -1;
                                    occurrences.forEach(function(occ) {
                                        if(occ.docIndex != docIndex) {
                                            docIndex = occ.docIndex;
                                            infos += '<h4>' + panel.getCorpus().getDocument(docIndex).getFullLabel() + ':</h4>';
                                        }
                                        infos += '<li>'+ occ.source.left+'<a href="http://www.geonames.org/' + occ.source.id + '" target="_blank" docIndex='+occ.docIndex +' offset='+occ.source.position+
                                            ' location='+occ.source.form + ' cityId=' + occ.source.id + ' coordinates=' + JSON.stringify(sourceCoordinates) + ' class="termLocationLink">' + occ.source.form + '</a> '+occ.source.right + ' [...] ' +
                                            occ.target.left+'<a href="http://www.geonames.org/' + occ.target.id + '" target="_blank" docIndex='+occ.docIndex+' offset='+occ.target.position+
                                            ' location='+occ.target.form+ ' cityId=' + occ.target.id + ' coordinates=' + JSON.stringify(targetCoordinates) + ' class="termLocationLink">' + occ.target.form + '</a> '+occ.target.right + '</li>';
                                    });
                                    var header = feature.get("text");
                                    infos += "</ul>";
                                    panel.getContentEl().setHtml('<h3>' + header + '</h3>' + infos);
                                    panel.getOverlay().setPosition(event.coordinate);
                                    var links = Ext.select('.termLocationLink');

//                                    function contextMenuListener(el) {
//                                        el.addEventListener( "contextmenu", function(e) {
//                                            e.preventDefault();
//                                            panel.openContextMenu(e.layerX, e.layerY, el);
//                                            return false;
//                                        }, false);
//                                    }

                                    links.elements.forEach(function(link) {
//                                        contextMenuListener(link);
                                        link.onmouseover = function(){panel.showTermInCorpus(link.getAttribute("docIndex"), link.getAttribute("offset"), link.getAttribute('location'))};
                                    });
                                }
                            });
                            foundFeature = true
                        } /*else if (feature.get("type") === "annotation" && !foundFeature) { //annotation
                            panel.setCurrentAnnotation(feature);
                            panel.getContentEl().setHtml('<textarea class="annotation-text" rows="5" cols="60">'+(feature.get("text")?feature.get("text"):"")+'</textarea>' +
                                '<button class="saveAnnotation">Save</button><button class="deleteAnnotation">Delete</button>');
                            panel.getOverlay().setPosition(event.coordinate);
                            panel.getContentEl().down('.deleteAnnotation').dom.onclick = function() {panel.deleteAnnotation()};
                            panel.getContentEl().down('.saveAnnotation').dom.onclick = function() {panel.saveAnnotation()};
                            panel.getContentEl().down('.annotation-text').dom.focus();
                        } */
                    }, this);
                }
            });

            // Layer for selected vector
            var selectedLayer = new ol.layer.Vector({
                map: map,
                source: new ol.source.Vector({
                    wrapX: false,
                    useSpatialIndex: false // optional, might improve performance
                }),
                zIndex: 10,
                selected: true,
                style: function(feature) {
                    if(feature.get("type") === "city")
                    {
                        return panel.cityStyleFunction(feature, map.getView().getResolution());
                    } else if(feature.get("type") === "connection") {
                        return panel.travelStyleFunction(feature, map.getView().getResolution());
                    }
                },
                updateWhileAnimating: true, // optional, for instant visual feedback
                updateWhileInteracting: true // optional, for instant visual feedback
            });

            // Layer for edit preview
            var previewLayer = new ol.layer.Vector({
                map: map,
                source: new ol.source.Vector({
                    wrapX: false,
                    useSpatialIndex: false // optional, might improve performance
                }),
                preview: true,
                zIndex: 15,
                id: "preview",
                style: function(feature) {
                    if(feature.get("type") === "city")
                    {
                        return panel.cityStyleFunction(feature, map.getView().getResolution());
                    } else if(feature.get("type") === "connection") {
                        return panel.travelStyleFunction(feature, map.getView().getResolution());
                    }
                },
                updateWhileAnimating: true, // optional, for instant visual feedback
                updateWhileInteracting: true // optional, for instant visual feedback
            });
            map.addLayer(previewLayer);

            // Add handler to update selected vector when mouse is moved
            map.on('pointermove', function(event) {
            	
            	// set cursoe to pointer if we have a feature
            	  var pixel = map.getEventPixel(event.originalEvent);
            	  var hit = map.hasFeatureAtPixel(pixel);
            	  map.getViewport().style.cursor = hit ? 'pointer' : '';
            	  
                if(!panel.getDrawMode()) {
                    selectedLayer.getSource().clear();
                    var coordinate = event.coordinate;
                    var pixel = event.pixel;
                    var features = map.getFeaturesAtPixel(pixel);
                    if(features) {
                        var i = 0;
                        while(features[i].get("selected")){
                            i++;
                            if (i === features.length) break;
                        }
                        if (i < features.length) {
                            var feature = features[i];

                            var baseTextStyle = {
                                font: '12px Calibri,sans-serif',
                                textAlign: 'center',
                                offsetY: -15,
                                fill: new ol.style.Fill({
                                    color: [0,0,0,1]
                                }),
                                stroke: new ol.style.Stroke({
                                    color: [255,255,255,0.5],
                                    width: 4
                                })
                            };

                            baseTextStyle.text = feature.get("text");

                            var textOverlayStyle = new ol.style.Style({
                                text: new ol.style.Text(baseTextStyle),
                                zIndex: 1
                            });

                            if (!(feature.get("type") === "annotation")) {
                                var selectedFeature = new ol.Feature({
                                    text: feature.get("text"),
                                    geometry: feature.getGeometry(),
                                    forms: feature.get("forms"),
                                    selected: true,
                                    coordinates: feature.get("coordinates"),
                                    width: feature.get("width"),
                                    visible: feature.get("visible"),
                                    color: feature.get("color"),
                                    type: feature.get("type"),
                                    confidence: feature.get("confidence"),
                                    source: feature.get("source"),
                                    target: feature.get("target"),
                                    cityId: feature.get("cityId")
                                });
                                selectedLayer.getSource().addFeature(selectedFeature);
                            }
                            var geometry = feature.getGeometry();
                            var point = event.coordinate;
                            var textFeature = new ol.Feature({
                                geometry: new ol.geom.Point(point),
                                selected: true
                            });
                            textFeature.setStyle(textOverlayStyle);
                            selectedLayer.getSource().addFeature(textFeature);

                            // Highlight all connection where selected city is source or target

                            if(feature.get("type") === "city") {
                                var layers = panel.getMap().getLayers();
                                layers.forEach(function (layer) {
                                    var layerId = layer.get("id");
                                    if (layerId && layerId.indexOf("connections") !== -1) {
                                        connections = layer.getSource().getFeatures();
                                        connections.forEach(function (connection) {
                                            if(connection.get("target") === feature.get("cityId") || connection.get("source") === feature.get("cityId")) {
                                                selectedLayer.getSource().addFeature(new ol.Feature({
                                                    geometry: connection.getGeometry(),
                                                    selected: true,
                                                    coordinates: connection.get("coordinates"),
                                                    width: connection.get("width"),
                                                    visible: connection.get("visible"),
                                                    color: connection.get("color"),
                                                    type: connection.get("type")
                                                }));
                                            }
                                        })
                                    }
                                })
                            }
                        }
                    }
                }
            });
            var source = new ol.source.Vector({wrapX: false});

            var annotations = new ol.layer.Vector({
                source: source,
                id: "annotations"
            });

            var annotationsObj = this.getAnnotations();
            if (annotationsObj) {
                var geojson = new ol.format.GeoJSON();
                var features = geojson.readFeatures(annotationsObj);
                source.addFeatures(features);
            }

            map.addLayer(annotations);

            this.setDrawInteraction(
                new ol.interaction.Draw({
                    source: source,
                    type: "Polygon",
                    freehand: true,
                    alias: "draw"
                })
            );
            this.getDrawInteraction().on('drawend', function (evt) {
                evt.feature.set('type', 'annotation');
//                panel.setCurrentAnnotation(evt.feature);
                panel.editAnnotation(evt.feature);
//                panel.getContentEl().setHtml('<textarea class="annotation-text" rows="5" cols="60"></textarea>'+
//                    '<button class="saveAnnotation">Save</button><button class="deleteAnnotation">Cancel</button>');
//                panel.getContentEl().down('.deleteAnnotation').dom.onclick = function() {panel.deleteAnnotation()};
//                panel.getContentEl().down('.saveAnnotation').dom.onclick = function() {panel.saveAnnotation()};
//                panel.getOverlay().setPosition(evt.feature.getGeometry().getLastCoordinate());
//                panel.getContentEl().down('.annotation-text').dom.focus();
            });

            /*
            map.getViewport().addEventListener('contextmenu', function (e) {
                e.preventDefault();
                panel.openContextMenu(e.layerX, e.layerY);
                return false;
            }, false);
            */

            this.setMap(map);

            var zoom = this.getTargetEl().down(".ol-zoom");
            Ext.create('Ext.Button', {
                glyph:'xf075@FontAwesome',
                cls: 'annotate',
                renderTo: this.getTargetEl().down(".ticker").parent(),
                tooltip: this.localize('annotateTip'),
                enableToggle: true,
                toggleHandler: function(cmp, state) {
                    if (state) {
                        this.getMap().addInteraction(this.getDrawInteraction());
                    } else {
                        this.getMap().removeInteraction(this.getDrawInteraction());
                    }
                },
                scope: this
            })

            // simulate adding a filter
            this.getDockedComponent('bottomToolbar').getComponent('addFilter').click();
        } else {
            Ext.defer(this.tryInit, 500, this); // try again in a half second
        }
    },
    
    editAnnotation: function(feature) {
    		var text = feature.get("text"), buttons = [];
    		Ext.Msg.show({
    			title: this.localize("editAnnotation"),
    			message: this.localize("editAnnotationMessage"),
    			buttons: Ext.Msg.OKCANCEL,
            multiline: true,
            value: text,
            scope: this,
            callback: function(btn, val) {
            		if (btn=='ok') {
            			if (val=="") {
            				this.getMap().getLayer("annotations").getSource().removeFeature(feature);
            			} else {
            				feature.set("text", val);
            			}
            	        var geojson  = new ol.format.GeoJSON;
            	        var features = this.getMap().getLayer("annotations").getSource().getFeatures();
            	        var transformedFeatures = [];
            	        var projection = this.getProjection();
            	        features.forEach(function (feature) {
            	            transformedFeature = feature.clone();
            	            transformedFeature.getGeometry().transform(projection?projection:'EPSG:3857', 'EPSG:3857');
            	            transformedFeatures.push(transformedFeature);
            	        })
            	        var json = geojson.writeFeatures(transformedFeatures);
            	        this.storeAnnotations(json);
            		}
            }
    		})

    },

    /*
    deleteAnnotation: function() {
        this.body.down('.ol-popup').down(".ol-popup-closer").dom.click();
        this.getMap().getLayer("annotations").getSource().removeFeature(this.getCurrentAnnotation());
        var geojson  = new ol.format.GeoJSON;
        var features = this.getMap().getLayer("annotations").getSource().getFeatures();
        var transformedFeatures = [];
        var projection = this.getProjection();
        features.forEach(function (feature) {
            transformedFeature = feature.clone();
            transformedFeature.getGeometry().transform(projection?projection:'EPSG:3857', 'EPSG:3857');
            transformedFeatures.push(transformedFeature);
        })
        var json = geojson.writeFeatures(transformedFeatures);
        this.storeAnnotations(json);
//        localStorage['annotations'] = JSON.stringify(json);
    },
    */

    storeAnnotations: function(json) {
        this.mask("storingAnnotations");
        var me = this;
        Ext.Ajax.request({
            url: this.getTromboneUrl(),
            params: {
                tool: 'resource.StoredResource',
                storeResource: JSON.stringify(json)
            },
            scope: this
        }).then(function(response) {
            me.unmask();
            var data = Ext.JSON.decode(response.responseText);
            me.setApiParam("annotationsId", data.storedResource.id);
            me.toastInfo(me.localize('annotationsUpdated'));
        }, function(response) {
            me.unmask();
            me.showResponseError(me.localize("annotationsUpdateFailed"), response);
        });
    },

    /*
    saveAnnotation: function() {
        this.getCurrentAnnotation().set("text", this.getContentEl().down('.annotation-text').dom.value);
        this.getOverlay().setPosition(undefined);
//        if(this.getDrawMode()) {
//            this.getDockedComponent('bottomToolbar').getComponent('annotate').click();
//        }
        var geojson  = new ol.format.GeoJSON;
        var features = this.getMap().getLayer("annotations").getSource().getFeatures();
        var transformedFeatures = [];
        var projection = this.getProjection();
        features.forEach(function (feature) {
            transformedFeature = feature.clone();
            transformedFeature.getGeometry().transform(projection?projection:'EPSG:3857', 'EPSG:3857');
            transformedFeatures.push(transformedFeature);
        })
        var json = geojson.writeFeatures(transformedFeatures);
        this.storeAnnotations(json);
//        localStorage['annotations'] = JSON.stringify(json);
    },
    */

    // Style for vector after animation
    travelStyleFunction: function(feature, resolution) {

        var stroke = new ol.style.Stroke({
            color: feature.get('color'),
            width: feature.get('width')
        });

        var styles = [
            new ol.style.Style({
                stroke: stroke
            })];

        // Add arrow at the end of vectors
        var geometry = feature.getGeometry();
        var end = geometry.getLastCoordinate();
        var beforeEnd = geometry.getCoordinateAt(0.9);
        var dx = end[0] - beforeEnd[0];
        var dy = end[1] - beforeEnd[1];
        var rotation = Math.atan2(dy, dx);

        var lineStr1 = new ol.geom.LineString([end, [end[0] - 10 * resolution, end[1] + 10 * resolution]]);
        lineStr1.rotate(rotation, end);
        var lineStr2 = new ol.geom.LineString([end, [end[0] - 10 * resolution, end[1] - 10 * resolution]]);
        lineStr2.rotate(rotation, end);

        styles.push(new ol.style.Style({
            geometry: lineStr1,
            stroke: stroke
        }));
        styles.push(new ol.style.Style({
            geometry: lineStr2,
            stroke: stroke
        }));

        return styles;
    },

    // Style for cities
    cityStyleFunction: function(feature, resolution) {
        if(feature.get("visible")) {
            var diameter = Math.PI * 2 * feature.get("width");
            var confArc = feature.get("confidence") ? diameter * feature.get("confidence") : diameter;
            return (new ol.style.Style({
                image: new ol.style.Circle({
                    radius: feature.get("width"),
                    fill: new ol.style.Fill({
                        color: feature.get("color")
                    }),
                    stroke: new ol.style.Stroke({
                        lineDash: [confArc, diameter - confArc],
                        color: 'rgb(255, 255, 255)',
                        width: 2
                    })
                })
            }));
        } else {
            return false;
        }
    },

    // Style for vector during animation
    animationStyleFunction: function (feature) {
        return [new ol.style.Style({
            stroke: new ol.style.Stroke({
                color: feature.get("color"),
                width: 5
            })
        }), new ol.style.Style({
            geometry: new ol.geom.Circle(feature.getGeometry().getFirstCoordinate()),
            stroke: new ol.style.Stroke({
                color: "white",
                width: 10
            })
        }), new ol.style.Style({
            geometry: new ol.geom.Circle(feature.getGeometry().getLastCoordinate()),
            stroke: new ol.style.Stroke({
                color: "white",
                width: 10
            })
        })];
    },

    showTermInCorpus: function(docIndex, position, location) {
        var term = Ext.create("Voyant.data.model.Context", {
            docIndex: docIndex,
            position: position,
            term: location
        });
        this.getApplication().dispatchEvent('termLocationClicked', this, [term]);
    },
   
    /**
     * Update all filters
     */
    reloadFilters: function() {
        this.getFilterWidgets().each(function(filter) {
        		filter.loadGeonames();
        }, this);
    },

    filterUpdate: function(filter) {
        var panel = this;
        filter.clearAnimation();

        var map = this.getMap(), color = filter.getColor();

        var citiesMaxCount = parseInt(this.getApiParam("citiesMaxCount"));

        // determine max rawFreq for sizing
        var max = 0;
        filter.getGeonames().eachCity(function(city) {if (city.rawFreq>max) {max=city.rawFreq;}}, this, citiesMaxCount);

        // generate cities
        var layerSource = map.getLayer(filter.getId()+"-cities").getSource();

        var hide = Ext.Array.from(this.getApiParam("hide"));

        layerSource.clear();
        var minPopulation = parseInt(this.getApiParam("minPopulation"));
        var citiesMinFreq = parseInt(this.getApiParam("citiesMinFreq"));
        var validCitiesHash = {};

        filter.getGeonames().eachCity(function(city) {

            if ((!minPopulation || city.population>=minPopulation) && (!citiesMinFreq || city.rawFreq>=citiesMinFreq)) {
                validCitiesHash[city.id]=true;
                var coordinates = [parseFloat(city.lng), parseFloat(city.lat)]
                var feature = new ol.Feature({

                    geometry: new ol.geom.Point(coordinates).transform(ol.proj.get('EPSG:4326'), panel.getProjection()?panel.getProjection():ol.proj.get('EPSG:3857')),
                    text: city.label + " ("+city.rawFreq+")",
                    description: city.label,
                    color: color,
                    selected: false,
                    width: 3+ (Math.sqrt(city.rawFreq/max)*20),
                    visible: true,
                    coordinates: coordinates,
                    forms: city.forms,
                    cityId: city.id,
                    type: "city",
                    confidence: city.confidence ? city.confidence*100 : undefined
                });
                layerSource.addFeature(feature);
            }
        }, this, citiesMaxCount);

        if (layerSource.getFeatures().length>0) {
            map.getView().fit(layerSource.getExtent());
        }

        map.getLayer(filter.getId()+"-cities").setVisible(Ext.Array.contains(hide, "cities")==false);

        layerSource = map.getLayer(filter.getId()+"-connections").getSource();
        layerSource.clear();

        if (Object.keys(validCitiesHash).length==0) {
            this.toastInfo("No cities available for current criteria.");
        }

        max = 0;
        filter.getGeonames().eachConnection(function(connection) {
            if (connection.rawFreq>max) {max=connection.rawFreq;}
        }, this);

        // we'll go through all collections to see if we have valid cities
        var counter = 0,
            maxConnectionsCount = parseInt(this.getApiParam('connectionsMaxCount')),
            connectionsMinFreq = parseInt(this.getApiParam('connectionsMinFreq'));
        filter.getGeonames().eachConnection(function(connection) {
            if ((!connectionsMinFreq || connection.rawFreq>=connectionsMinFreq) && (!maxConnectionsCount || counter<maxConnectionsCount) && connection.source.id in validCitiesHash && connection.target.id in validCitiesHash && (!minPopulation || (connection.source.population>=minPopulation && connection.target.population>=minPopulation))) {
                var arcGenerator = new arc.GreatCircle(
                    {x: connection.source.lng, y: connection.source.lat},
                    {x: connection.target.lng, y: connection.target.lat});
                var arcLine = arcGenerator.Arc(100, {offset: 100});
                var label = connection.source.label+" -> "+connection.target.label + " ("+connection.rawFreq+")"
                arcLine.geometries.forEach(function(geometry) {
                    var line = new ol.geom.LineString(geometry.coords);
                    line.transform(ol.proj.get('EPSG:4326'), panel.getProjection()?panel.getProjection():ol.proj.get('EPSG:3857'));
                    var feature = new ol.Feature({
                        geometry: line,
                        text: label,
                        color: color,
                        width: 3+ (Math.sqrt(connection.rawFreq/max)*10),
                        source: connection.source.id,
                        target: connection.target.id,
                        type: "connection"
                    });
                    layerSource.addFeature(feature);
                }, this);
                counter++;
            }
        }, this);

        map.getLayer(filter.getId()+"-connections").setVisible(Ext.Array.contains(hide, "connections")==false)

        filter.animate();

    },
    
    handleSingleClick: function(event) {
        var features = this.getMap().getFeaturesAtPixel(event.pixel);
        if (features) {
        		for (var i=0; i<features.length; i++) {
        			var feature = features[i];
        			var featureType = feature.get("type")
        			if (featureType=='city') {
        				this.handleLocationClick(event, feature);
        				break;
        			} else if (featureType=="connection") {
        				this.handleConnectionClick(feature, event.pixel[0], event.pixel[1]);
        				break;
        			}else if (featureType=="annotation") {
        				this.editAnnotation(feature);
        				break;
        			}
        		}
        }

    },
    
    handleConnectionClick: function(feature, x, y) {
		Ext.create('Ext.menu.Menu', {
			items: [{
    				text: this.localize("viewConnections"),
    				tooltip: this.localize("viewConnectionsTip"),
    				glyph: 'xf02d@FontAwesome',
    				handler: function() {
    					this.mask("loadingConnections");
    					var me = this;
    		            Ext.Ajax.request({
    		                url: this.getTromboneUrl(),
    		                params: {
    		                    tool: 'corpus.Dreamscape',
    		                    corpus: this.getCorpus().getId(),
    		                    suppressLocations: true,
    		                    suppressConnections: true,
    		                    sourceId: feature.get("source"),
    		                    targetId: feature.get("target"),
    		                    context: 2,
    		                    limit: 25
    		                },
    		                scope: this
    		            }).then(function(response) {
	                		me.unmask();
    		                var data = Ext.JSON.decode(response.responseText);
    		                if (data && data.dreamscape && data.dreamscape.connectionOccurrences) {
    		                		var out = "<table style='font-size:smaller'>";
    		                		data.dreamscape.connectionOccurrences.connectionOccurrences.forEach(function(occurrence) {
    		                			out+="<tr><td style='text-align: right'>"+occurrence.source.left+"</td><td class='keyword' style='text-align: center'>"+
    		                				occurrence.source.term+"</td><td>"+occurrence.source.right+"</td><td>…</td><td style='text-align: right'>"+occurrence.target.left+"</td><td class='keyword' style='text-align: center'>"+
    		                				occurrence.target.term+"</td><td>"+occurrence.target.right+"</td></tr>";
    		                		})
    		                		out += "</table>";
    		                		Ext.Msg.alert(me.localize("occurrences"), out);
    		                }
    		            }, function(response) {
	                		me.unmask();
                			return me.showError(me.localize("occurrencesNotLoaded"))	    		                			
    		            });
    				},
    				scope: this
				}]
		}).showAt(x, y)
    },
    
    
    handleLocationClick: function(event, feature) {
		var currentGeonameId = feature.get("cityId");
		var e = Ext.create('Ext.menu.Menu', {
			items: [{
    				text: this.localize("viewOccurrences"),
    				tooltip: this.localize("viewOccurrencesTip"),
    				glyph: 'xf02d@FontAwesome',
    				handler: function() {
    					this.mask("loadingOccurrences");
    					var me = this;
    		            Ext.Ajax.request({
    		                url: this.getTromboneUrl(),
    		                params: {
    		                    tool: 'corpus.Dreamscape',
    		                    corpus: this.getCorpus().getId(),
    		                    suppressLocations: true,
    		                    suppressConnections: true,
    		                    suppressConnectionOccurrences: true,
    		                    locationId: currentGeonameId,
    		                    limit: 25
    		                },
    		                scope: this
    		            }).then(function(response) {
	                		me.unmask();
    		                var data = Ext.JSON.decode(response.responseText);
    		                if (data && data.dreamscape && data.dreamscape.occurrences) {
    		                		var out = "<table style='font-size:smaller'>";
    		                		data.dreamscape.occurrences.occurrences.forEach(function(occurrence) {
    		                			out+="<tr><td style='text-align: right'>"+occurrence.left+"</td><td class='keyword' style='text-align: center'>"+occurrence.term+"</td><td>"+occurrence.right+"</td></tr>";
    		                		})
    		                		out += "</table>";
    		                		Ext.Msg.alert(me.localize("occurrences"), out);
    		                }
    		            }, function(response) {
	                		me.unmask();
                			return me.showError(me.localize("occurrencesNotLoaded"))	    		                			
    		            });
    				},
    				scope: this
				},{
    				text: this.localize("openInVoyant"),
    				tooltip: this.localize("openInVoyantTip"),
    				glyph: 'xf08e@FontAwesome',
    				handler: function() {
    					this.dispatchEvent('termsClicked', this, feature.get("forms"));
    				},
    				scope: this
				},{
    				text: this.localize("editLocation"),
    				tooltip: this.localize("editLocationTip"),
    				glyph: 'xf124@FontAwesome',
    				handler: function() {
    					this.mask("loadingAlternatives");
    					var me = this;
    		            Ext.Ajax.request({
    		                url: this.getTromboneUrl(),
    		                params: {
    		                    tool: 'util.Geonames',
    		                    query: feature.get("forms")
    		                },
    		                scope: this
    		            }).then(function(response) {
	                		me.unmask();
    		                var data = Ext.JSON.decode(response.responseText);
    		                if (data && data.geonames && data.geonames.locations) {
    		                		var items = []; 		 
    		                		Object.values(data.geonames.locations.locations).forEach(function(location) {
    		                			items.push({
    		                				boxLabel: location.label+" ("+location.population+")",
    		                				name: 'id',
    		                				inputValue: location.id,
    		                				checked: location.id==currentGeonameId
    		                			})
    		                		})
    		                		if (items.length==0) {
    		                			return me.showError(me.localize("editLocationNoLocationsFound"))
    		                		} else if (items.length==1 && items[0].inputValue==currentGeonameId) {
    		                			return me.showError(me.localize("editLocationNoAlternativesFound"))	    		                			
    		                		}
	    		                	Ext.create('Ext.window.Window', {
								title: me.localize("editLocation"),
								modal: true,
								items: {
									xtype: 'form',
									items: [{
								        xtype: 'radiogroup',
								        columns: 1,
								        vertical: true,
								        items: items
								    }],
									listeners: {
										afterrender: function(form) {
										},
										scope: this
									},
									buttons: [{
										text: me.localize("cancel"),
							            ui: 'default-toolbar',
						                glyph: 'xf00d@FontAwesome',
						        			flex: 1,
						        			handler: function(btn) {
						        				btn.up('window').destroy();
						        			}
									},{
										text: me.localize("confirmTitle"),
										glyph: 'xf00c@FontAwesome',
					            			flex: 1,
					            			handler: function(btn) {
					            				var newId = btn.up("form").getValues().id;
					            				if (newId && newId!=currentGeonameId) {
					    							var config = {};
					    							config[currentGeonameId] = newId;
					    							me.appendOverrides(config);
					            				}
						        				btn.up('window').destroy();
					            			}
									}]
								}
							}).show();
    		                }
    		            }, function(response) {
	                		me.unmask();
    		                me.showResponseError(me.localize("editLocationServerError"), response);

    		            });
    				},
    				scope: this
			},{
    				text: this.localize("removeLocation"),
    				tooltip: this.localize("removeLocationTip"),
    				glyph: 'xf00d@FontAwesome',
    				handler: function() {
    					Ext.Msg.confirm(
	    					this.localize("removeLocationConfirmTitle"),
	    					this.localize("removeLocationConfirm"),
	    					function(btn) {
	    						if (btn=="yes") {
	    							var config = {};
	    							config[currentGeonameId] = "";
	    							this.appendOverrides(config);
	    						}
	    					},
	    					this
    					)
    				},
    				scope: this
			}    				

			],
			listeners: {
				focusleave : function(menu) {
					menu.destroy();
				}
			}
		}).showAt(event.pixel[0], event.pixel[1])    		
    },
    
    appendOverrides: function(config, isAll) {
    		// if we have overridesId defined but we haven't loaded all IDs yet
    		if (this.getApiParam("overridesId") && !isAll) {
    			var me = this;
    			Ext.Ajax.request({
                url: this.getTromboneUrl(),
                params: {
                    tool: 'resource.StoredResource',
                    retrieveResourceId: this.getApiParam("overridesId")
                },
                scope: this
            }).then(function(response) {
                var data = Ext.JSON.decode(response.responseText);
                if (data && data.storedResource && data.storedResource.resource) {
                		var newConfig = Ext.decode(data.storedResource.resource);
                		Ext.apply(newConfig, config);
                		// call again, this time specifying that we have all
                		me.appendOverrides(newConfig, true);
                } else {
                    me.showResponseError(me.localize("overidesRetrieveError"), response);
                }
            	
            }, function(response) {
                me.showResponseError(me.localize("overidesRetrieveError"), response);
            });
    		} else {
    			var me = this;
    			Ext.Ajax.request({
                url: this.getTromboneUrl(),
                params: {
                    tool: 'resource.StoredResource',
                    storeResource: Ext.encode(config)
                },
                scope: this
            }).then(function(response) {
                var data = Ext.JSON.decode(response.responseText);
                if (data && data.storedResource && data.storedResource.id) {
                    me.setApiParam("overridesId", data.storedResource.id);
                    me.reloadFilters();
                } else {
                    me.showResponseError(me.localize("overidesRetrieveError"), response);
                }
            }, function(response) {
                me.showResponseError(me.localize("overidesStorageError"), response);
            });
    		}
    }    

});

Ext.define('Voyant.widget.GeonamesFilter', {
    extend: 'Ext.Button',
    xtype: 'geonamesfilter',
    mixins: ['Voyant.util.Localization'],
    statics: {
        i18n: {
            filter: 'Filter',
            authorLabel: 'authors',
            titleLabel: 'titles',
            keywordLabel: 'full text',
            pubDateLabel: "dates",
            loading: "Loading geographical information…",
//            loadedAll: "All available cities and connections have been loaded (see options under the <i>Show</i> menu in the toolbar).",
//            loadedSome: "Loaded {currentCitiesCount} of {totalCitiesCount} available locations and {currentConnectionsCount} of {totalConnectionsCount} available connections (see options under the <b>Show</b> menu in the toolbar).",
//            disclaimer: "Please note that geographical information is generated automatically and may be inaccurate (<a href='{url}' target='_blank'>more information</a>).</p>",
            disclaimer: "This is an experimental tool and the accuracy of the data is variable (<a href='{url}' target='_blank'>see help</a>).",
            noDataForField: "No data seems to be available for this field so it will be disabled.",
            close: "Close"
        }
    },
    config: {
        corpus: undefined, // used by querysearchfields among other things
        geonames: undefined,
        color: "#f00",
        timeout: undefined,
        currentConnectionOccurrence: undefined,
        animationLayer: undefined,
        millisPerAnimation: 2000,
        stepByStepMode: false,
        keepAnimationInFrame: false
    },
    constructor: function(config) {
        config = config || {};
        var geonames = new Voyant.data.util.Geonames({
            corpus: config.corpus
        });
        this.setGeonames(geonames);
        var me = this;
        Ext.applyIf(config,
            {
                tooltip: this.localize('filterTip'),
                style: "background-color: white; color: "+config.color,
                glyph: 'xf0b0@FontAwesome',
                menu: {
                    defaults: {
                        xtype: 'querysearchfield',
                        labelWidth: 60,
                        labelAlign: 'right',
                        width: 250,
                        maxWidth: 250,
                        padding: 2,
                        listeners: { // don't set scope to this so that load keeps the widget scope
                            change: function(cmp, vals) {
                                me.loadGeonames();
                            },
                            load: function(store, records, success, operation) {
                                if (records.length==0 && operation.getParams().query=="") {
                                    Ext.Msg.show({
                                        buttonText: {ok: me.localize('close')},
                                        icon: Ext.MessageBox.INFO,
                                        message: me.localize("noDataForField"),
                                        buttons: Ext.Msg.OK
                                    });
                                    this.disable();
                                }
                            }
                        }

                    },
                    items: [{
                        fieldLabel: this.localize('authorLabel'),
                        tokenType: 'author',
                        itemId: 'author'
                    },{
                        fieldLabel: this.localize('titleLabel'),
                        itemId: 'title',
                        tokenType: 'title'
                    },{
                        fieldLabel: this.localize('keywordLabel')
                    },{
                        xtype: 'multislider',
                        fieldLabel: this.localize('pubDateLabel'),
                        itemId: 'pubDate',
                        tokenType: 'pubDate',
                        values: [0,1],
                        hidden: true,
                        listeners: {
                            afterrender: function(cmp) {
                                this.getCorpus().getCorpusTerms().load({
                                    params: {
                                        tokenType: 'pubDate',
                                        limit: 1,
                                        sort: "TERMASC"
                                    },
                                    callback: function(records) {
                                        if (records.length==0) {
                                            // no pubDate, see if we can parse from titles
                                            var vals = [];
                                            this.getCorpus().each(function(doc) {
                                                var val = parseInt(doc.getTitle());
                                                if (isNaN(val)==false) {
                                                    vals.push(parseInt(doc.getTitle()))
                                                }
                                            }, this);
                                            if (vals.length>1) {
                                                cmp.setMinValue(vals[0]);
                                                cmp.setMaxValue(vals[vals.length-1]);
                                                cmp.tokenType = 'title';
                                                cmp.suspendEvent('changecomplete');
                                                cmp.setValue([vals[0], vals[vals.length-1]]);
                                                cmp.resumeEvent('changecomplete');
                                                cmp.setVisible(true);
                                            }
                                        } else {
                                            cmp.setMinValue(parseInt(records[0].getTerm()))
                                            this.getCorpus().getCorpusTerms().load({
                                                params: {
                                                    tokenType: 'pubDate',
                                                    limit: 1,
                                                    sort: "TERMDESC"
                                                },
                                                callback: function(records) {
                                                    cmp.setMaxValue(parseInt(records[0].getTerm())),
                                                        cmp.setVisible(true);
                                                },
                                                scope: this
                                            });
                                        }
                                    },
                                    scope: this
                                })
                            },
                            changecomplete: function(cmp, newVal) {
                                this.loadGeonames();
                            },
                            scope: this
                        }
                    }, {
                        xtype: 'fieldset',
                        title: "Animation",
                        items: [{
                            xtype: 'container',
                            layout: 'hbox',
                            defaults: {
                                xtype: 'button',
                                text: "",
                                ui: 'default-toolbar'
                            },
                            items: [{
                                glyph: 'xf048@FontAwesome', // step back
                                handler: function() {
                                    this.getAnimationLayer().getSource().clear();
                                    this.clearAnimation();
                                    var currentConnectionOccurrence = this.getCurrentConnectionOccurrence();
                                    currentConnectionOccurrence = this.getGeonames().getConnectionOccurrence(currentConnectionOccurrence ? currentConnectionOccurrence.index-1 : 0);
                                    this.setCurrentConnectionOccurrence(currentConnectionOccurrence);
                                    this.animate();
                                },
                                scope: this
                            },{
                                glyph: 'xf04c@FontAwesome', // play
                                handler: function(cmp) {
                                    if (cmp.getGlyph().glyphConfig=="xf04b@FontAwesome") {
                                        this.clearAnimation();
                                        cmp.setGlyph(new Ext.Glyph('xf04c@FontAwesome'));
                                        this.setStepByStepMode(false);
                                        this.animate();
                                    } else {
                                        cmp.setGlyph(new Ext.Glyph('xf04b@FontAwesome'));
//                                         this.clearAnimation();
//                                         this.getAnimationLayer().getSource().clear();
                                        this.setStepByStepMode(true);
                                    }
//                                     this.setCurrentConnectionOccurrence(this.getGeonames().getConnectionOccurrence(0));
                                },
                                scope: this

                            },{
                                glyph: 'xf051@FontAwesome', // step forward
                                handler: function() {
                                    this.getAnimationLayer().getSource().clear();
                                    this.clearAnimation();
                                    var currentConnectionOccurrence = this.getCurrentConnectionOccurrence();
                                    currentConnectionOccurrence = this.getGeonames().getConnectionOccurrence(currentConnectionOccurrence ? currentConnectionOccurrence.index+1 : 0);
                                    this.setCurrentConnectionOccurrence(currentConnectionOccurrence);
                                    this.animate();
                                },
                                scope: this
                            }, {
                                xtype: 'tbtext',
                                text: "&nbsp;&nbsp;"
                            }, {
                                glyph: 'xf01e@FontAwesome', // reset
                                handler: function() {
                                    this.getAnimationLayer().getSource().clear();
                                    this.clearAnimation();
                                    this.setCurrentConnectionOccurrence(this.getGeonames().getConnectionOccurrence(0));
                                    this.animate();
                                },
                                scope: this
                            }]
                        }, {
                            xtype: 'checkbox',
                            checked: false,
                            handler: function(item, checked) {
                                me.setKeepAnimationInFrame(checked);
                            },
                            boxLabel: "Keep animation in frame"
                        }]

                    },{
                        xtype: "container",
                        text: "&nbsp;"
                    },{
                        xtype: 'button',
                        text: 'remove',
                        scope:this,
                        //margin: 3,
                        handler: function(cmp) {
                            // assumes coupling with dreamscape
                            this.fireEvent("removeFilterWidget", this);
                            cmp.ownerCt.ownerCmp.destroy();
                        }
                    }]
                }
            });
        this.callParent([config]);
        this.on("afterrender", function(cmp) {
            cmp.getTargetEl().down('.x-btn-glyph').setStyle('color', this.getColor());
            var panel = cmp.up("panel")
            panel.mask(cmp.localize("loading"));
            cmp.loadGeonames().then(function(geonames) {
                panel.unmask();
                if (panel.getFilterWidgets().getCount()==1) {
                    var currentCitiesCount = geonames.getCitiesCount(),
                        totalCitiesCount = geonames.getTotalCitiesCount(),
                        currentConnectionsCount = geonames.getConnectionsCount(),
                        totalConnectionsCount = geonames.getTotalConnectionsCount();
                    message = cmp.localize(currentCitiesCount==totalCitiesCount && currentConnectionsCount==totalConnectionsCount ? "loadedAll" : "loadedSome") +
                        "<br><br>"+cmp.localize("disclaimer");
                    panel.toastInfo({
                        autoCloseDelay: 5000,

                        closable: true,
                        maxWidth: '90%',
                        html: new Ext.Template(cmp.localize("disclaimer")).apply({
                            currentCitiesCount: currentCitiesCount,
                            totalCitiesCount: totalCitiesCount,
                            currentConnectionsCount: currentConnectionsCount,
                            totalConnectionsCount: totalConnectionsCount,
                            url: panel.getBaseUrl()+"docs/#!/guide/dreamscape"
                        })

                    });
                }
            })
        }, this);
    },
    loadGeonames: function(params) {
        var me = this;
        params = params || {};

        // add queries
        var queries  = [];
        this.query('querysearchfield').forEach(function(querysearchfield) {
            var id = querysearchfield.getItemId();
            querysearchfield.getValue().forEach(function(query) {
                queries.push(id+":"+query);
            })
        });
        var pubDateSlider = this.down("multislider");
        if (pubDateSlider && pubDateSlider.isVisible()) {
            var vals = pubDateSlider.getValue();
            queries.push(pubDateSlider.tokenType+":["+vals[0]+"-"+vals[1]+"]")
        }
        if (queries.length>0 && !("query" in params)) {params.query=queries;}

        // add params from tool
        var panel  = this.up('panel');
        if (panel && panel.getApiParams) {
            Ext.applyIf(params, panel.getApiParams(["minPopulation","connectionsMaxCount","connectionsMinFreq","source","overridesId","filterHasLowerCaseForm","filterIsPersonName","preferredCoordinates"]))
        }

        this.mask("…");
        var geonames = this.getGeonames();
        return geonames.load(params).then(function() {
            me.unmask();
            me.fireEvent("filterUpdate", me, geonames);
            return geonames;
        }, function() {
            me.unmask();
        });
    },

    clearAnimation: function() {
        if (this.getTimeout()) {clearTimeout(this.getTimeout());}
    },

    animate: function() {
        var panel = this.up('panel');
        if (!panel) {return;}
        this.clearAnimation();
        var currentConnectionOccurrence = this.getCurrentConnectionOccurrence();
        if (currentConnectionOccurrence) {
            var animationLayer = this.getAnimationLayer();
            if (!animationLayer) {
                animationLayer = this.up('panel').getMap().getLayer(this.getId()+"-animation");
                this.setAnimationLayer(animationLayer);
            }
            if (animationLayer) {
                var features = animationLayer.getSource().getFeatures();
                if (features.length==0) {
                    if (!panel) {return;}
                    var connectionsLayerSource = panel.getMap().getLayer(this.getId()+"-connections").getSource();
                    var connectionsFeatures = connectionsLayerSource.getFeatures();
                    var hasMatch = false;
                    for (var i=0, len=connectionsFeatures.length; i<len; i++) {
                        if (currentConnectionOccurrence.source.id==connectionsFeatures[i].get("source") && currentConnectionOccurrence.target.id==connectionsFeatures[i].get("target")) {
                            hasMatch = connectionsFeatures[i];
                            break;
                        }
                    }
                    if (hasMatch) {
                        // check to see if we need to reposition the map, use displayed feature as basis for extent
                        if(this.getKeepAnimationInFrame()) {
                            var map = panel.getMap();
                            var extent = panel.getMap().getView().calculateExtent(map.getSize());
                            var isVisible = ol.extent.containsExtent(extent, hasMatch.getGeometry().getExtent());
                            if(!isVisible) {
                                newExtent = ol.extent.extend(extent, hasMatch.getGeometry().getExtent());
                                map.getView().fit(newExtent);
                            }
                        }

                        var arcGenerator = new arc.GreatCircle(
                            {x: currentConnectionOccurrence.source.lng, y: currentConnectionOccurrence.source.lat},
                            {x: currentConnectionOccurrence.target.lng, y: currentConnectionOccurrence.target.lat});
                        var arcLine = arcGenerator.Arc(100, {offset: 100});
                        var label = currentConnectionOccurrence.source.label+" -> "+currentConnectionOccurrence.target.label;
                        arcLine.geometries.forEach(function(geometry) {
                            var line = new ol.geom.LineString([]); // can be empty since we generate a line during next animate call
                            line.transform(ol.proj.get('EPSG:4326'), panel.getProjection()?panel.getProjection():ol.proj.get('EPSG:3857'));
                            var feature = new ol.Feature({
                                geometry: line,
                                allcoords: geometry.coords ,
                                text: label,
                                color: this.getColor(),
                                width: 5,
                                start: new Date().getTime()
                            });
                            animationLayer.getSource().addFeature(feature);
                        }, this);

                        var ticker = panel.body.down(".ticker");

                        panel.body.down(".ticker").setHtml(
                            currentConnectionOccurrence.source.left+" <span class='keyword'>"+currentConnectionOccurrence.source.term+"</span> "+currentConnectionOccurrence.source.right+" … "+
                            currentConnectionOccurrence.target.left+" <span class='keyword'>"+currentConnectionOccurrence.target.term+"</span> "+currentConnectionOccurrence.target.right
                        );
                        return this.setTimeout(setTimeout(this.animate.bind(this), 1));

                    } else {
                        if (!this.getGeonames()) {return;}
                        currentConnectionOccurrence = this.getGeonames().getConnectionOccurrence(currentConnectionOccurrence.index+1);
                        this.setCurrentConnectionOccurrence(currentConnectionOccurrence);
                        animationLayer.getSource().clear();
                        return this.setTimeout(setTimeout(this.animate.bind(this), 1));

                    }
                } else if (features.length>0) {

                    var coords = features[0].getGeometry().getCoordinates(),
                        allcoords = features[0].get("allcoords");
                    if (coords.length<allcoords.length) {
                        var elapsed = new Date().getTime()-features[0].get('start'),
                            len = elapsed*allcoords.length/this.getMillisPerAnimation()
                        line = new ol.geom.LineString(allcoords.slice(0,len));
                        line.transform(ol.proj.get('EPSG:4326'), panel.getProjection()?panel.getProjection():ol.proj.get('EPSG:3857'));
                        features[0].set("geometry", line);
                        return this.setTimeout(setTimeout(this.animate.bind(this), 25));

                    }
                    // if we get here, all features should have been changed
                    currentConnectionOccurrence = this.getGeonames().getConnectionOccurrence(currentConnectionOccurrence.index+1);
                    if(!this.getStepByStepMode()) {
                        this.setCurrentConnectionOccurrence(currentConnectionOccurrence);
                        animationLayer.getSource().clear();
                        return this.setTimeout(setTimeout(this.animate.bind(this), 1));
                    }
                }
            }
        } else {
            if (!this.getGeonames()) {return;}
            currentConnectionOccurrence = this.getGeonames().getConnectionOccurrence(0);
            if (currentConnectionOccurrence) {
                this.setCurrentConnectionOccurrence(currentConnectionOccurrence);
                return this.setTimeout(setTimeout(this.animate.bind(this), 1));
            }
        }
    }
});
