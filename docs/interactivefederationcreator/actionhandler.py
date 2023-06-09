# ComDeX Action Handler Tool

# Version: 0.6.1
# Author: Nikolaos Papadakis 
# Requirements:
# - Python 3.7 or above
# - Shapely library
# - paho-mqtt library

# For more information and updates, visit: [https://github.com/SAMSGBLab/ComDeX]

from pickle import TRUE
import sys
import json
import getopt
import paho.mqtt.client as mqtt 
import time
import socket
import logging
import urllib.parse
import threading
import multiprocessing
import pprint
import re
import ast
import datetime
import shapely.geometry as shape_geo
import urllib.request



#default values of mqtt broker to communicate with
default_broker_adress='localhost'
default_broker_port=1026
default_ngsild_context="https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"

#global advertisement flag (to avoid for now passing it in every function)
singleidadvertisement=False

#TO DO convert these globals to nonlocals 
#exists=False
exists_topic=''
full_data=''

    


#Function: post_entity
#Description: This function is used to create a new NGSI-LD entity in the ComDeX node.
#Parameters:
#- data: The data of the entity to be created.
#- my_area: The area or domain of the entity.
#- broker: The name or IP address of the broker.
#- port: The port number of the broker.
#- qos: The quality of service level for message delivery.
#- my_loc: The location of the broker (used for advanced advertisements with geoqueries).
#- bypass_existence_check (optional): Flag to bypass the existence check of the entity (default: 0).
#- client (optional): MQTT client object (default: mqtt.Client(clean_session=True)).
# Returns: None
def post_entity(data,my_area,broker,port,qos,my_loc,bypass_existence_check=0,client=mqtt.Client(clean_session=True)):
    

    global singleidadvertisement

    client.loop_start()     
    if 'type' in data:
        typee=str(data['type'])
    else:
        print("Error, ngsi-ld entity without a type \n")
        sys.exit(2)
    if 'id' in data:  
        id=str(data['id'])
    else:
        print("Error, ngsi-ld entity without a id \n")
        sys.exit(2)
    if '@context' in data:
        if( str(type(data["@context"]))=="<class 'str'>"):
            context=data['@context'].replace("/", "§")
        else:
            context=data['@context'][0].replace("/", "§")

        
        
    else:    
        print("Error, ngsi-ld entity without context \n")
        sys.exit(2)
    if 'location' in data:
        location=data['location'] 
    else:
        location=''       
    
    big_topic=my_area+'/entities/'+context+'/'+typee+'/LNA/'+id     

 
    check_topic='+/entities/'+context+'/'+typee+'/+/'+id+'/#' 
    print("Show me the check topic" + check_topic)
    print("Checking existence of entity...")
    
   
    if(bypass_existence_check==0):
        if (check_existence(broker,port,check_topic)!=False):
            print("Error entity with this id already exists, did you mean to patch?")
            return

    #check for remote existance maybe in the future
      
    ################### CREATE SMALL TOPICS!!!!!!!!!!!!!!!#######################
    for key in data.items():
        if key[0]!="type" and key[0]!="id" and key[0]!='@context':
            
            small_topic=my_area+'/entities/'+context+'/'+typee+'/LNA/'+id+'/'+key[0]
            #print(small_topic)
            print("Publishing message to subtopic")    
            
            client.publish(small_topic,str(key[1]),retain=True,qos=qos)
            
            curr_time=str(datetime.datetime.now())
            time_rels = { "createdAt": [curr_time],"modifiedAt": [curr_time] }

            small_topic=my_area+'/entities/'+context+'/'+typee+'/LNA/'+id+'/'+ key[0]+"_timerelsystem_CreatedAt"
                
            client.publish(small_topic,str(time_rels["createdAt"]),retain=True,qos=qos)
            
            small_topic=my_area+'/entities/'+context+'/'+typee+'/LNA/'+id+'/'+ key[0]+"_timerelsystem_modifiedAt"
            client.publish(small_topic,str(time_rels["modifiedAt"]),retain=True,qos=qos)     

    ############################################################################
    check_topic2="provider/+/+/"+my_area+'/'+context+'/'+typee+'/'
    
    if(singleidadvertisement==False):
        special_context_provider_broadcast= 'provider/' + broker + '/' +str(port) + '/'+my_area+'/' + context + '/' +typee
    else:
        special_context_provider_broadcast= 'provider/' + broker + '/' +str(port) + '/'+my_area+'/' + context + '/' +typee +'/'+id
        bypass_existence_check=1
    
    if(bypass_existence_check==1):
        client.publish(special_context_provider_broadcast,"Provider Message: { CreatedAt:" + str(time_rels["createdAt"]) +",location:" + str(my_loc)+"}" ,retain=True,qos=2)
        print(special_context_provider_broadcast) 
    elif(check_existence(broker,port,special_context_provider_broadcast)==False):
        client.publish(special_context_provider_broadcast,"Provider Message: { CreatedAt:" + str(time_rels["createdAt"]) +",location:" + str(my_loc)+"}" ,retain=True,qos=2)

        print("Publishing message to provider table")  
        #old logging of published messages
        #logger = logging.getLogger()
        #handler = logging.FileHandler('logfile_advertisement_published.log')
        #logger.addHandler(handler)
        #logger.error(time.time_ns()/(10**6))

        print(special_context_provider_broadcast)        
    client.loop_stop()


#Description: This function checks if an entity or advertisement already exists inside the broker.
#Parameters:
#- broker: The name or IP address of the broker.
#- port: The port number of the broker.
#- topic: The topic name or identifier of the message to check.
#Returns:
#- True if the entity/advertisement exists in the broker, False otherwise.

def check_existence(broker,port,topic):
    run_flag=TRUE
    exists=False
    expires=1
    def on_connect(client3, userdata, flags, rc):
        print("Connected for existence check with result code "+str(rc))

    # The callback for when a PUBLISH message is received from the server.
    def on_message(client3, userdata, msg):
        global exists_topic
        nonlocal exists
        nonlocal expires
        exists=True
        exists_topic=msg.topic
        expires-=1
    
    client3 = mqtt.Client()   
    
    client3.on_connect = on_connect
    client3.on_message = on_message

    
    client3.connect(broker, port)
    client3.loop_start()
    client3.subscribe(topic,qos=1)

    start=time.perf_counter()
    try:
        while run_flag:
            tic_toc=time.perf_counter()
            if (tic_toc-start) > expires:
                run_flag=False
    except:
        pass
    #time.sleep(1)
    client3.loop_stop()  
    #print(exists)
    return exists    


# Function: GET
# Description: This function is used to retrieve entities from the ComDeX node, similar to the NGSI-LD GET entities operation.
# Parameters:
#   - broker: The name or IP address of the broker.
#   - port: The port number of the broker.
#   - topics: A list of topics to subscribe to.
#   - expires: The expiration time in seconds.
#   - qos: The quality of service level for message delivery.
#   - limit (optional): The maximum number of entities to retrieve (default: 2000).
# Returns:
#   - A list of received messages (entities) ordered via their id.

def GET(broker, port, topics, expires, qos, limit=2000):
    run_flag = True
    messagez = []
    messages_by_id = {}

    # The callback for when a PUBLISH message is received from the server.
    def on_message(client, userdata, msg):
        nonlocal messagez
        nonlocal expires
        nonlocal messages_by_id
        nonlocal limit
        if msg.retain == 1:
            initial_topic = msg.topic.split('/')
            id = initial_topic[-2]
            messages_by_id.setdefault(id, []).append(msg)
            if len(messages_by_id) == limit + 1:
                expires -= 10000000
            else:
                messagez.append(msg)
                expires += 0.5

    # Create an MQTT client
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker, port)

    client.loop_start()

    # Subscribe to the specified topics
    for topic in topics:
        client.subscribe(topic, qos)

    start = time.perf_counter()

    try:
        while run_flag:
            tic_toc = time.perf_counter()
            if tic_toc - start > expires:
                run_flag = False
    except:
        pass

    client.loop_stop()

    # Return the received messages (entities)
    return messagez


# Function: recreate_single_entity
# Description: This function recreates a single entity from the received messages based on the specified query conditions.
# This is possible because each entity has a unique message id, which is used as the catalyst for the entity reconstruction from
# its various attribute messages
# Parameters:
#   - messagez: List of received messages (entities).
#   - query: Query condition to filter the entities (optional, default: '').
#   - topics: Topic filters to apply (optional, default: '').
#   - timee: Time condition to filter the entities (optional, default: '').
#   - georel: Geo-relation condition to filter the entities (optional, default: '').
#   - geometry: Geometry type for the geospatial condition (optional, default: '').
#   - coordinates: Coordinates for the geospatial condition (optional, default: '').
#   - geoproperty: Geospatial property for the geospatial condition (optional, default: '').
#   - context_given: Context value for entity comparison (optional, default: '').
# Returns: None

def recreate_single_entity(messagez, query='', topics='', timee='', georel='', geometry='', coordinates='', geoproperty='', context_given=''):
    query_flag_passed = False
    subqueries_flags = {}
    default_context = "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"

    # Extract initial topic information
    initial_topic = (messagez[0].topic).split('/')
    id = initial_topic[-2]
    typee = initial_topic[-4]
    context = initial_topic[-5]
    context = context.replace("§", "/")
    context_text = context
    contextt = []
    contextt.append(context_text)

    # Add default context if it differs from the specified context
    if context_text != default_context:
        contextt.append(default_context)

    # Initialize data dictionary with ID and type
    data = {}
    data['id'] = id
    data['type'] = typee

    # Check if a specific context is given for comparison
    if context_given == '+':
        with urllib.request.urlopen(context_text) as url:
            data_from_web = json.loads(url.read().decode())
        try:
            data['type'] = data_from_web["@context"][typee]
        except:
            dummy_command = "This is a dummy command for except"

    if query == '':
        for msg in messagez:
            attr_str = msg.payload
            attr_str = attr_str.decode(encoding='UTF-8', errors='strict')
            attr_str = attr_str.replace("\'", "\"")
            data2 = json.loads(attr_str)
            topic = (msg.topic).split('/')

            # Check geospatial condition if specified
            if georel != '':
                if topic[-1] == geoproperty:
                    geo_type = str(data2["value"]["type"])
                    geo_coord = str(data2["value"]["coordinates"])
                    geo_ok = 0

                    geo_type = geo_type.replace(" ", "")
                    geo_coord = geo_coord.replace(" ", "")
                    coordinates = coordinates.replace(" ", "")

                    geo_entity = shape_geo.shape((data2["value"]))

                    if geometry == "Point":
                        query_gjson = shape_geo.Point(json.loads(coordinates))
                    elif geometry == "LineString":
                        query_gjson = shape_geo.LineString(json.loads(coordinates))
                    elif geometry == "Polygon":
                        query_gjson = shape_geo.Polygon(json.loads(coordinates))
                    elif geometry == "MultiPoint":
                        query_gjson = shape_geo.MultiPoint(json.loads(coordinates))
                    elif geometry == "MultiLineString":
                        query_gjson = shape_geo.MultiLineString(json.loads(coordinates))
                    elif geometry == "MultiPolygon":
                        query_gjson = shape_geo.MultiPolygon(json.loads(coordinates))

                    # Check specific georelation condition
                    if georel == "equals":
                        if geo_entity.equals(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif georel == "within":
                        if geo_entity.within(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif georel == "intersects":
                        if geo_entity.intersects(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif re.search("near;", georel):
                        near_query = georel.split(';')
                        near_operator = re.findall('[><]|==|>=|<=', near_query[1])
                        near_geo_queries = (re.split('[><]|==|>=|<=', near_query[1]))

                        if str(near_geo_queries[0]) == "maxDistance":
                            if str(near_operator[0]) == "==":
                                if geo_entity.distance(query_gjson) > float(near_geo_queries[1]):
                                    return
                        elif str(near_geo_queries[0]) == "minDistance":
                            if str(near_operator[0]) == "==":
                                if geo_entity.distance(query_gjson) < float(near_geo_queries[1]):
                                    return
                    elif georel == "contains":
                        if geo_entity.contains(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif georel == "disjoint":
                        if geo_entity.disjoint(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif georel == "overlaps":
                        if geo_entity.overlaps(query_gjson):
                            geo_ok = 1
                        else:
                            return

            # Check topic filters if specified
            if topics != '' and topics != "#":
                if topic[-1] in topics:
                    data[topic[-1]] = data2
                if topic[-1].endswith("_CreatedAt") or topic[-1].endswith("_modifiedAt"):
                    if timee != '':
                        time_topic = (topic[-1].split('_timerelsystem_'))
                        if context_given == '+':
                            try:
                                time_topic[-2] = data_from_web["@context"][time_topic[-2]]
                            except:
                                dummy_command = "This is a dummy command for except"

                        data[time_topic[-2]][time_topic[-1]] = data2

            else:
                if topic[-1].endswith("_CreatedAt") or topic[-1].endswith("_modifiedAt"):
                    if timee != '':
                        time_topic = (topic[-1].split('_timerelsystem_'))
                        if context_given == '+':
                            try:
                                time_topic[-2] = data_from_web["@context"][time_topic[-2]]
                            except:
                                dummy_command = "This is a dummy command for except"

                        data[time_topic[-2]][time_topic[-1]] = data2
                else:
                    data[topic[-1]] = data2

        data['@context'] = contextt

        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        print(json_data)

    elif query != '':
        logical_operators = re.findall('[;|()]', query)
        queries_big = re.split(('[;|()]'), query)

        for msg in messagez:
            attr_str = msg.payload
            attr_str = attr_str.decode(encoding='UTF-8', errors='strict')
            attr_str = attr_str.replace("\'", "\"")
            data2 = json.loads(attr_str)
            topic = (msg.topic).split('/')

            # Check geospatial condition if specified
            if georel != '':
                if topic[-1] == geoproperty:
                    geo_type = str(data2["value"]["type"])
                    geo_coord = str(data2["value"]["coordinates"])
                    geo_ok = 0

                    geo_type = geo_type.replace(" ", "")
                    geo_coord = geo_coord.replace(" ", "")
                    coordinates = coordinates.replace(" ", "")

                    geo_entity = shape_geo.shape((data2["value"]))

                    if geometry == "Point":
                        query_gjson = shape_geo.Point(json.loads(coordinates))
                    elif geometry == "LineString":
                        query_gjson = shape_geo.LineString(json.loads(coordinates))
                    elif geometry == "Polygon":
                        query_gjson = shape_geo.Polygon(json.loads(coordinates))
                    elif geometry == "MultiPoint":
                        query_gjson = shape_geo.MultiPoint(json.loads(coordinates))
                    elif geometry == "MultiLineString":
                        query_gjson = shape_geo.MultiLineString(json.loads(coordinates))
                    elif geometry == "MultiPolygon":
                        query_gjson = shape_geo.MultiPolygon(json.loads(coordinates))

                    # Check specific georelation condition
                    if georel == "equals":
                        if geo_entity.equals(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif georel == "within":
                        if geo_entity.within(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif georel == "intersects":
                        if geo_entity.intersects(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif re.search("near;", georel):
                        near_query = georel.split(';')
                        near_operator = re.findall('[><]|==|>=|<=', near_query[1])
                        near_geo_queries = (re.split('[><]|==|>=|<=', near_query[1]))

                        if str(near_geo_queries[0]) == "maxDistance":
                            if str(near_operator[0]) == "==":
                                if geo_entity.distance(query_gjson) > float(near_geo_queries[1]):
                                    return
                        elif str(near_geo_queries[0]) == "minDistance":
                            if str(near_operator[0]) == "==":
                                if geo_entity.distance(query_gjson) < float(near_geo_queries[1]):
                                    return
                    elif georel == "contains":
                        if geo_entity.contains(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif georel == "disjoint":
                        if geo_entity.disjoint(query_gjson):
                            geo_ok = 1
                        else:
                            return
                    elif georel == "overlaps":
                        if geo_entity.overlaps(query_gjson):
                            geo_ok = 1
                        else:
                            return

            # Allowing combination of logical queries
            for query2 in queries_big:
                operator = re.findall('[><]|==|>=|<=', query2)
                queries = (re.split('[><]|==|>=|<=', query2))
                subqueries_flags.setdefault(queries[0], False)

                if queries[0] == topic[-1]:

                    if str(operator[0]) == "==":

                        if isinstance(data2["value"], list):
                            for data3 in data2["value"]:
                                if data3 == queries[1]:
                                    subqueries_flags[queries[0]] = True

                        elif data2["value"] == queries[1]:
                            subqueries_flags[queries[0]] = True
                    elif queries[1].isnumeric():
                        if str(operator[0]) == ">":
                            if float(data2["value"]) > float(queries[1]):
                                subqueries_flags[queries[0]] = True
                        elif str(operator[0]) == "<":
                            if float(data2["value"]) < float(queries[1]):
                                subqueries_flags[queries[0]] = True
                        elif str(operator[0]) == "<=":
                            if float(data2["value"]) <= float(queries[1]):
                                subqueries_flags[queries[0]] = True
                        elif str(operator[0]) == ">=":
                            if float(data2["value"]) >= float(queries[1]):
                                subqueries_flags[queries[0]] = True

            # Check topic filters if specified
            if topics != '' and topics != "#":
                if topic[-1] in topics:
                    data[topic[-1]] = data2
                if topic[-1].endswith("_CreatedAt") or topic[-1].endswith("_modifiedAt"):
                    if timee != '':
                        time_topic = (topic[-1].split('_timerelsystem_'))
                        if context_given == '+':
                            try:
                                time_topic[-2] = data_from_web["@context"][time_topic[-2]]
                            except:
                                dummy_command = "This is a dummy command for except"

                        data[time_topic[-2]][time_topic[-1]] = data2

            else:
                if topic[-1].endswith("_CreatedAt") or topic[-1].endswith("_modifiedAt"):
                    if timee != '':
                        time_topic = (topic[-1].split('_timerelsystem_'))
                        if context_given == '+':
                            try:
                                time_topic[-2] = data_from_web["@context"][time_topic[-2]]
                            except:
                                dummy_command = "This is a dummy command for except"

                        data[time_topic[-2]][time_topic[-1]] = data2
                else:
                    data[topic[-1]] = data2

        data['@context'] = contextt

        l = 0
        full_logical_equation = []
        subqueries_flags.pop('', None)
        for results in subqueries_flags.values():
            full_logical_equation.append(str(results))
            if l < (len(logical_operators)):
                if logical_operators[l] != '':
                    if logical_operators[l] == ";":
                        full_logical_equation.append('and')
                    elif logical_operators[l] == "|":
                        full_logical_equation.append('or')
                    else:
                        full_logical_equation.append(logical_operators[l])
                    if l + 1 < (len(logical_operators) - 1):
                        while logical_operators[l + 1] != ';' and logical_operators[l + 1] != '|':
                            l = l + 1
                            full_logical_equation.append(logical_operators[l])

            l = l + 1

        query_flag_passed = eval(' '.join(full_logical_equation))
        if query_flag_passed == True:
            json_data = json.dumps(data, indent=4, ensure_ascii=False)
            print(json_data)
        
                 



        
# Function: recreate_multiple_entities
# Description: This function recreates multiple entities from the received messages based on the specified query conditions. It basically calls the 
# recreate single entity command, over a list of MQTT messages, based on their id.
# Parameters:
#   - messagez: List of received messages (entities).
#   - query: Query condition to filter the entities (optional, default: '').
#   - topics: Topic filters to apply (optional, default: '').
#   - timee: Time condition to filter the entities (optional, default: '').
#   - limit: Maximum number of entities to recreate (optional, default: 2000).
#   - georel: Geo-relation condition to filter the entities (optional, default: '').
#   - geometry: Geometry type for the geospatial condition (optional, default: '').
#   - coordinates: Coordinates for the geospatial condition (optional, default: '').
#   - geoproperty: Geospatial property for the geospatial condition (optional, default: '').
#   - context_given: Context value for entity comparison (optional, default: '').
# Returns: None

def recreate_multiple_entities(messagez, query='', topics='', timee='', limit=2000, georel='', geometry='', coordinates='', geoproperty='', context_given=''):
    messages_by_id = {}

    # Separate each message by ID to recreate using the recreate_single_entity function
    for message in messagez:
        initial_topic = (message.topic).split('/')
        id = initial_topic[-2]
        messages_by_id.setdefault(id, []).append(message)

    # Iterate over single entities and recreate them using recreate_single_entity function
    for single_entities in messages_by_id.values():
        recreate_single_entity(single_entities, query, topics, timee, georel, geometry, coordinates, geoproperty, context_given)

        # Countdown pagination limit
        limit = limit - 1
        if limit == 0:
            break




        

# Function: multiple_subscriptions
# Description: This function sets up multiple subscriptions based on the specified flags and parameters.
# Parameters:
#   - entity_type_flag: Flag indicating whether entity type is specified.
#   - watched_attributes_flag: Flag indicating whether watched attributes are specified.
#   - entity_id_flag: Flag indicating whether entity ID is specified.
#   - area: Area value for the subscriptions.
#   - context: Context value for the subscriptions.
#   - truetype: Entity type value for the subscriptions.
#   - true_id: Entity ID value for the subscriptions.
#   - expires: Expiration time for the subscriptions.
#   - broker: MQTT broker address.
#   - port: MQTT broker port.
#   - qos: Quality of Service level for the subscriptions.
#   - watched_attributes: List of watched attributes for the subscriptions.
# Returns: None

def multiple_subscriptions(entity_type_flag, watched_attributes_flag, entity_id_flag, area, context, truetype, true_id, expires, broker, port, qos, watched_attributes):
    topic = []
    
    if entity_type_flag and watched_attributes_flag and entity_id_flag:
        # Subscribe to topics based on entity type, watched attributes, and entity ID
        for attr in watched_attributes:
            topic.append(area + '/entities/' + context + '/' + truetype + '/+/' + true_id + '/' + attr)
    
    elif entity_type_flag and entity_id_flag:
        # Subscribe to topics based on entity type and entity ID
        topic.append(area + '/entities/' + context + '/' + truetype + '/+/' + true_id + '/#')
    
    elif watched_attributes_flag and entity_id_flag:
        # Subscribe to topics based on watched attributes and entity ID
        for attr in watched_attributes:
            topic.append(area + '/entities/' + context + '/+/+/' + true_id + '/' + attr)
    
    elif entity_type_flag and watched_attributes_flag:
        # Subscribe to topics based on entity type and watched attributes
        for attr in watched_attributes:
            topic.append(area + '/entities/' + context + '/' + truetype + '/+/+/' + attr)
    
    elif entity_type_flag:
        # Subscribe to topics based on entity type
        topic.append(area + '/entities/' + context + '/' + truetype + '/#')
    
    elif entity_id_flag:
        # Subscribe to topics based on entity ID
        topic.append(area + '/entities/' + context + '/+/+/' + true_id + '/#')
    
    elif watched_attributes_flag:
        # Subscribe to topics based on watched attributes
        for attr in watched_attributes:
            topic.append(area + '/entities/' + context + '/+/+/+/' + attr)
    
    else:
        print("Something has gone wrong, program did not find the topics to subscribe to!")
        sys.exit(2)
    
    # Call the subscribe function with the generated topics
    subscribe(broker, port, topic, expires, qos, context_given=context)
    



# Function: subscribe
# Description: This function subscribes to MQTT topics and handles the received messages.
# Parameters:
#   - broker: MQTT broker address.
#   - port: MQTT broker port.
#   - topics: List of topics to subscribe to.
#   - expires: Expiration time for the subscriptions.
#   - qos: Quality of Service level for the subscriptions.
#   - context_given: Context value for entity comparison.
# Returns: None

def subscribe(broker, port, topics, expires, qos, context_given):
    run_flag = True

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))

    # The callback for when a PUBLISH message is received from the server.
    def on_message(client, userdata, msg):
        # Perform operations on received message
        if msg.payload.decode() != '':
            stamp = time.time_ns() / (10 ** 6)
            initial_topic = (msg.topic).split('/')
            ids = initial_topic[-2]
            tmp = msg.payload
            attr_str = msg.payload.decode(encoding='UTF-8', errors='strict')
            attr_str = attr_str.replace("\'", "\"")
            data2 = json.loads(attr_str)

            #Older logging messages for experiments
            # Logging the message arrival time difference
            #handler = logging.FileHandler('logfile_mqtt_notification_arrived.log')
            # logger.addHandler(handler)
            # logger.error(str(stamp - float(data2["value"])))

        if msg.payload.decode() != '':
            messagez = []
            messagez.append(msg)
            if msg.topic.endswith("_CreatedAt") or msg.topic.endswith("_modifiedAt"):
                # Do nothing for special system time topics (ignore them, no temporal subscriptions)
                do_nothing = 1
            else:
                # Recreate single entity based on the received message
                recreate_single_entity(messagez, timee=0, context_given=context_given)
        else:
            print("\n Message on topic:" + msg.topic + ", was deleted")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to MQTT broker
    client.connect(broker, port, keepalive=expires)

    client.loop_start()
    for topic in topics:
        client.subscribe(topic, qos)
        #print("Subscribing to topic: " +topic)

    start = time.perf_counter()

    try:
        while run_flag:
            tic_toc = time.perf_counter()
            if (tic_toc - start) > expires:
                run_flag = False
    except:
        pass

    print("Subscriptions expired, exiting.....")
    # Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a manual interface.
    client.loop_stop()

# Function: subscribe_for_advertisement_notification
# Description: This function sets up subscriptions for advertisement notifications based on the specified flags and parameters, basically to
# find if a new advertisement of interest arrives while an intersted subscriber is active (so that the subscriber while also connect to the
# new advertised source "on the fly").
# Parameters:
#   - broker: MQTT broker address.
#   - port: MQTT broker port.
#   - topics: List of topics to subscribe to for advertisement notifications.
#   - expires: Expiration time for the subscriptions.
#   - qos: Quality of Service level for the subscriptions.
#   - entity_type_flag: Flag indicating whether entity type is specified.
#   - watched_attributes_flag: Flag indicating whether watched attributes are specified.
#   - entity_id_flag: Flag indicating whether entity ID is specified.
#   - watched_attributes: List of watched attributes for the subscriptions.
#   - true_id: Entity ID value for the subscriptions.
# Returns: None

def subscribe_for_advertisement_notification(broker, port, topics, expires, qos, entity_type_flag, watched_attributes_flag, entity_id_flag, watched_attributes, true_id):
    run_flag = True
    advertisement_exists = {}
    jobs_to_terminate = {}

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(client, userdata, flags, rc):
        print("Connected for advertisement notification with result code " + str(rc))

    # The callback for when a PUBLISH message is received from the server.
    def on_message(client, userdata, msg):
        #print("MESSAGE advert HERE")
        print(msg.payload.decode())
        if msg.payload.decode() != '':
            # Calculate at which time an interested subscriber received an advertisement message
            # that did not exist prior to its subscription
            nonlocal advertisement_exists

            stamp = time.time_ns() / (10 ** 6)

            initial_topic = (msg.topic).split('/')
            broker_remote = initial_topic[1]
            port_remote = int(initial_topic[2])
            area_remote = initial_topic[3]
            context = initial_topic[4]
            truetype = initial_topic[5]

            topic2 = "provider/" + broker_remote + '/' + str(port_remote) + '/' + area_remote + '/' + context + '/' + truetype
            attr_str = topic2
            print(attr_str)

            if topic2 in advertisement_exists.keys():
                print("advertisement_already_exists")
                return ()
            else:
                print("found_brand_new_advertisement")
                advertisement_exists.setdefault(topic2, [])

            topic = []

            context_providers_addresses = []
            context_providers_ports = []
            context_providers_areas = []
            number_of_threads = 1

            context_providers_addresses.append(initial_topic[1])
            context_providers_ports.append(initial_topic[2])
            context_providers_areas.append(initial_topic[3])

            nonlocal jobs_to_terminate
            jobs = []
            for i in range(0, number_of_threads):
                #print("How many threads???")
                process = multiprocessing.Process(
                    target=multiple_subscriptions,
                    args=(
                        entity_type_flag, watched_attributes_flag, entity_id_flag,
                        context_providers_areas[i], context, truetype, true_id, expires,
                        context_providers_addresses[i], int(context_providers_ports[i]), qos, watched_attributes
                    )
                )
                jobs.append(process)

            print(jobs)
            jobs_to_terminate.setdefault(topic2, jobs)
            jobs_to_terminate[topic2] = jobs
            for j in jobs:
                print(j)
                j.start()

        else:
            initial_topic = (msg.topic).split('/')
            broker_remote = initial_topic[1]
            port_remote = int(initial_topic[2])
            area_remote = initial_topic[3]
            context = initial_topic[4]
            truetype = initial_topic[5]

            topic2 = "provider/" + broker_remote + '/' + str(port_remote) + '/' + area_remote + '/' + context + '/' + truetype
            print("Advertisement Deleted")
            advertisement_exists.pop(topic2, 1)
            print(jobs_to_terminate)
            for j in jobs_to_terminate[topic2]:
                print(j)
                j.kill()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker, port, keepalive=expires)

    client.loop_start()
    for topic in topics:
        client.subscribe(topic, qos)
        print("Subscribing to topic: " + topic)

    start = time.perf_counter()

    try:
        while run_flag:
            tic_toc = time.perf_counter()
            if (tic_toc - start) > expires:
                run_flag = False
    except:
        pass

    print("Subscriptions expired, exiting.....")
    client.loop_stop()


# Function: clear_retained
# Description: This function clears the retained messages on the specified topic(s).
# Parameters:
#   - broker: MQTT broker address.
#   - port: MQTT broker port.
#   - retained: Single topic or list of topics to clear retained messages from.
# Returns: None

def clear_retained(broker, port, retained):
    run_flag = True
    expires = 0.5

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))

    # The callback for when a PUBLISH message is received from the server.
    def on_message(client, userdata, msg):
        nonlocal expires

        if (msg.retain == 1):
            # Publish a null message to clear the retained message
            client2.publish(msg.topic, None, 0, True)
            print("Clearing retained on topic -", msg.topic)
            expires += 0.1

    client = mqtt.Client()
    client2 = mqtt.Client()

    client.on_connect = on_connect
    client2.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker, port)
    client2.connect(broker, port)
    client.loop_start()
    client2.loop_start()
    client.subscribe(retained, qos=1)

    start = time.perf_counter()
    try:
        while run_flag:
            tic_toc = time.perf_counter()
            if (tic_toc - start) > expires:
                run_flag = False
    except:
        pass

    client.loop_stop()
    client2.loop_stop()


#debug functions to see mqtt broker communication
############
# Function: on_message
# Description: Callback function called when a message is received from the MQTT broker.
# Parameters:
#   - client: MQTT client instance.
#   - userdata: Custom userdata provided by the client.
#   - message: MQTT message object containing the received message.
# Returns: None

def on_message(client, userdata, message):
    # Print the received message payload, topic, QoS, and retain flag
    print("message received ", str(message.payload.decode("utf-8")))
    print("message topic=", message.topic)
    print("message qos=", message.qos)
    print("message retain flag=", message.retain)

# Function: on_log
# Description: Callback function called when a log message is generated by the MQTT client.
# Parameters:
#   - client: MQTT client instance.
#   - userdata: Custom userdata provided by the client.
#   - level: Log level of the message.
#   - buf: Log message string.
# Returns: None

def on_log(client, userdata, level, buf):
    # Print the log message
    print("log: ", buf)
#########################################   




def usage():
    print("\nUsage:")
    print("python actionhandler.py [options]\n")
    
    print("Options:")
    print("-h, --help                Show this help message and exit")
    print("-c, --command             Provide the command, possible commands include [POST/entities,POST/Subscriptions,DELETE/entities/,PATCH/entities/,GET/entities/,entityOperations/delete,entityOperations/create,entityOperations/update,entityOperations/upsert]")
    print("-f, --file                Specify the file to be used as input to the command")
    print("-b, --broker_address      Specify the address of the MQTT broker of the ComDeX node")
    print("-p, --port                Specify the port number of the MQTT broker of the ComDeX node")
    print("-q, --qos                 Specify the Quality of Service level (0, 1, or 2) to be used for the specified command")
    print("-H, --HLink               Specify the HLink, 'context link' to be used for the GET request")
    print("-A, --singleidadvertisement Specify if single ID advertisement is to be used (use 1 for True), default is false")
    
    print("\nExample:")
    print("python3 actionhandler.py -c POST/entities -f entity.ngsild -b localhost -p 1026 -q 1 -H HLink -A 0\n")




#ComDeX is an ngsild compliant "broker" that utilises a running MQTT broker
#Here is the main function where different functions are called mainly based on the selected by the user command.
def main(argv):
    #command line arguments check
    try:
        opts, args = getopt.getopt(argv,"hc:f:b:p:l:q:H:A:",["command=","file=","broker_adress=","port=","qos=","HLink=","singleidadvertisement="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    command=''
    file=''
    HLink=''
    qos=0
    Forwarding=1
    broker=default_broker_adress
    port=default_broker_port
    expires= 3600
    global singleidadvertisement
    my_area="unknown_area"
    my_loc="unknown_location"
    #parse command line args
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-c", "--command"):
            command = arg
        elif opt in ("-f", "--file"):
            file = arg
        elif opt in ("-b", "--broker_adress"):
            broker = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-q", "--qos"):
            qos = int(arg)
            if (qos < 0 or qos > 2):
                print("Invalid Mqtt qos")
                sys.exit(2)
        elif opt in ("-H", "--HLink"):
            HLink = arg
        elif opt in ("-A", "-singleidadvertisement"):
            if (arg=="1"):
                singleidadvertisement=True
           
    #Broker location awareness
    locations_file = open("broker_location_awareness.txt", "a+")
    locations_file.close()  
    locations_file2= open("broker_location_awareness.txt", "r")
    contents = locations_file2.read()
    try:
        location_awareness = ast.literal_eval(contents)

        locations_file2.close()       
        area=broker+":"+str(port)+":area"
        loc=broker+":"+str(port)+":loc"
        if area in location_awareness:
            my_area=location_awareness[area]
        if loc in location_awareness:
            my_loc=location_awareness[loc]
    except:
        print("No area or location was detected/stored for this broker")        

    if(command==''):
        print("No command found, exiting...")
        sys.exit()

    #POST ENTITIES    
    if command=="POST/entities":
        print("creating new instance")
        client = mqtt.Client(clean_session=True) #create new instance
        client.on_message=on_message #attach function to callback
        print("connecting to broker")
        client.connect(broker,port) #connect to broker
        if file=='':
            usage()
            sys.exit(2)
        print("\n ngsild Post entity command detected \n")
        with open(file) as json_file:
            try:
                data = json.load(json_file)
            except:
                print("Can't parse the input file, are you sure it is valid json?")
                sys.exit(2)    
        post_entity(data,my_area,broker,port,qos,my_loc,0,client)
       
    #CREATION OF SUBSRIPTIONS        
    elif command=='POST/Subscriptions':
        truetype=''
        true_id=''
        trueid2=''
        topic=[]
        entity_type_flag=False
        watched_attributes_flag=False
        entity_id_flag=False
        watched_attributes=''
        server_area_flag=False
        if file=='':
            usage()
            sys.exit(2)
        print("\nngsild Post Subscription command detected\n")
        with open(file) as json_file:
            try:
                data = json.load(json_file)
            except:
                print("Can't parse the input file, are you sure it is valid json?")
                sys.exit(2)
                
            if 'type' in data:
                typee=str(data['type'])
                if typee!="Subscription":
                    print('Subscription has invalid type: '+typee)
                    sys.exit(2)
            else:
                print("Error, ngsi-ld Subscription without a type \n")
                sys.exit(2)
            if 'id' in data:
                id=str(data['id'])
            else:
                print("Error, ngsi-ld Subscription without a id \n")
                sys.exit(2)
            if '@context' in data:
                if( str(type(data["@context"]))=="<class 'str'>"):
                    context=data['@context'].replace("/", "§")
                else:
                    context=data['@context'][0].replace("/", "§")
            else:
                context='+'    
            if 'entities' in data:
                info_entities=data['entities'][0]
                if 'type' in info_entities:
                    truetype=str(info_entities["type"])
                    entity_type_flag=True
                if 'id' in info_entities:
                    true_id=str(info_entities["id"])
                    entity_id_flag=True   
            if 'watchedAttributes' in data:
                watched_attributes=data['watchedAttributes']
                print(watched_attributes)
                watched_attributes_flag=True
                if watched_attributes == None:
                    print("Watched attributes without content, exiting....")
                    sys.exit(2)
            if 'expires' in data:
                expires=int(data['expires'])         

            if(entity_type_flag==False and watched_attributes_flag==False and entity_id_flag==False):    
                print("Error, ngsi-ld subscription without information about topics, exiting.... \n")
                sys.exit(2)    


            big_topic=my_area+'/Subscriptions/'+context+'/'+typee+'/LNA/'+id     

            print("creating new instance")
            client1 = mqtt.Client(clean_session=True) #create new instance
            client1.on_message=on_message #attach function to callback
            print("connecting to broker")
            client1.connect(broker,port) #connect to broker
            client1.loop_start() #start the loop
            print("Publishing message to topic")
            client1.publish(big_topic,str(data),qos=qos)
            client1.loop_stop() #stop the loop

            area=[]

            if 'area' in data:
                area=data['area']
            else:
                area.append('+')

            if(truetype==''):
                truetype2='#'
                truetype='+'
            else:
                truetype2=truetype
            
            if(true_id==''):
                trueid2='#'
            else:
                trueid2=true_id    

            messages_for_context=[]
            check_top=[]

            
            for z in area: 
                if(singleidadvertisement==False):
                    check_topic2="provider/+/+/"+z+'/'+context+'/'+truetype2
                else:
                    check_topic2="provider/+/+/"+z+'/'+context+'/'+truetype+'/'+trueid2  
                print(check_topic2)
                
                check_top.append(check_topic2)
            
            print(check_top)
            subscribe_for_advertisement_notification(broker,port,check_top,expires,qos,entity_type_flag,watched_attributes_flag,entity_id_flag,watched_attributes,true_id)
             
                #print(data)

    elif(re.search("DELETE/entities/",command)):
        #print("FOUND THE DELETE COMMAND!!!")
        command_parts = command.split("/")
        
        if(len(command_parts)<5):
            if (HLink==''):
                HLink='+'
                context_flag=False
            else:
                HLink=HLink.replace("/", "§")
        
            id=command_parts[2]
            typee='+'
            top=my_area+'/entities/'+HLink+'/+/+/'+id+'/#' 
            print(top)
            if (check_existence(broker,port,top)==False):
                print("Entity with this id doesn't exist, no need for deletion")
                sys.exit(2) 
            
            clear_retained(broker,port,top)

            
            tp=exists_topic.split("/")[-4]
            top_check_for_advert=my_area+'/entities/'+HLink+'/'+tp+'/+/+/#'
            print(top_check_for_advert)
            #remove from context source tables
            if(singleidadvertisement==False):
                special_context_provider_broadcast= 'provider/' + broker + '/' +str(port) + '/'+my_area+'/' + HLink + '/' +tp
                if (check_existence(broker,port,top_check_for_advert )==False):
                
                    special_context_provider_broadcast= 'provider/' + broker + '/' +str(port) + '/' + my_area + '/' + HLink + '/+'
                    clear_retained(broker,port,special_context_provider_broadcast)
            else:
                special_context_provider_broadcast= 'provider/' + broker + '/' +str(port) + '/'+my_area+'/' + HLink + '/' +tp +'/'+id
                clear_retained(broker,port,special_context_provider_broadcast)          
        else:
            if (HLink==''):
                HLink='+'
                context_flag=False
            else:
                HLink=HLink.replace("/", "§")
            if(command_parts[3]!="attrs"):
                print("Please check that you are writing the delete atrr command correctly")
                sys.exit(2)    
            command_parts = command.split("/")
            id=command_parts[2]
            top=my_area+'/entities/'+HLink+'/+/+/'+id+'/'+command_parts[4] 
            print(top)  
            clear_retained(broker,port,top)
              

    elif(re.search("PATCH/entities/",command)):
        print("Patch entity command detected")
        
        if (HLink==''):
            HLink='+'
            context_flag=False
        else:
            HLink=HLink.replace("/", "§")
        command_parts = command.split("/")
        
        if(len(command_parts)<5):
            print("Please check that you are writing the patch command correctly")
            sys.exit(2)

        if(command_parts[3]!="attr"):
            print("Please check that you are writing the patch command correctly")
            sys.exit(2)

        id=command_parts[2] 

        #do patch to id
        if command_parts[4]=='':
            print("Patch to id")
            if file=='':
                usage()
                sys.exit(2)
            
            typee='+'
            check_topic='+/entities/'+HLink+'/+/+/'+id+'/#'
            if (check_existence(broker,port,check_topic)==False):
                print("Error entity with this id doesn't exist, did you mean to POST?")
                sys.exit(2)
            
            with open(file) as json_file:
                try:
                    data = json.load(json_file)
                except:
                    print("Can't parse the input file, are you sure it is valid json?")
                    sys.exit(2)
                
                client = mqtt.Client(clean_session=True) #create new instance
                client.on_message=on_message #attach function to callback
                print("connecting to broker")
                client.connect(broker,port) #connect to broker
                client.loop_start() #start 

                tp=exists_topic.split("/")[-4]
                print(tp)

                for key in data.items():
                    if key[0]!="type" and key[0]!="id" and key[0]!='@context':
                        
                        small_topic=my_area+'/entities/'+HLink+'/'+tp+'/LNA/'+id+'/'+key[0]
                        #print(small_topic)
                        print("Publishing message to subtopic")    
                        client.publish(small_topic,str(key[1]),retain=True,qos=qos)

                        curr_time=str(datetime.datetime.now())
                        time_rels = { "createdAt": [curr_time],"modifiedAt": [curr_time] }                      
                        small_topic=my_area+'/entities/'+HLink+'/'+tp+'/LNA/'+id+'/'+ key[0]+"_timerelsystem_modifiedAt"
                        client.publish(small_topic,str(time_rels["modifiedAt"]),retain=True,qos=qos)     
                client.loop_stop()          

        #do patch to attribute    
        else:
            print("Patch to attribute")
            attrib=command_parts[4]
            if file=='':
                usage()
                sys.exit(2)
            
            typee='+'
            check_topic=my_area+'/entities/'+HLink+'/+/+/'+id+'/#'
            if (check_existence(broker,port,check_topic)==False):
                print("Error entity with this id doesn't exist, did you mean to POST?")
                sys.exit(2)
            
            with open(file) as json_file:

                try:
                    data = json.load(json_file)
                except:
                    print("Can't parse the input file, are you sure it is valid json?")
                    sys.exit(2)
                
                client = mqtt.Client(clean_session=True) #create new instance
                client.on_message=on_message #attach function to callback
                print("connecting to broker")
                client.connect(broker,port) #connect to broker
                client.loop_start() #start 

                tp=exists_topic.split("/")[-4]
                loc=exists_topic.split("/")[-3]
                print(tp)

                for key in data.items():
                    small_topic=my_area+'/entities/'+HLink+'/'+tp+'/'+loc+'/'+id+'/'+key[0]
                    print("Publishing message to subtopic")    
                    client.publish(small_topic,str(key[1]),retain=True,qos=qos)

                    curr_time=str(datetime.datetime.now())                     
                    time_rels = { "createdAt": [curr_time],"modifiedAt": [curr_time] } 
                    small_topic=my_area+'/entities/'+HLink+'/'+tp+'/LNA/'+id+'/'+ key[0]+"_timerelsystem_modifiedAt"
                    client.publish(small_topic,str(time_rels["modifiedAt"]),retain=True,qos=qos)     
                    client.loop_stop() 



    elif(re.search("GET/entities/",command)): 
        context_flag=True
        entity_id_flag=False
        entity_id_pattern_flag=False
        entity_type_flag=False
        entity_attrs_flag=False
        entity_query_flag=False
        context_flag=True  
        topic=[]
        area=[]
        typee_multi=[]
        timee=''
        limit=1800
        id='+'
        attrs='#'
        query=''
        geometry=''
        georel=''
        coordinates=''
        geoproperty='location'  #default value for ngsild
        geovar_count=0
        

        print("Get entity command found")

        if HLink=='':
            HLink='+'
            context_flag=False
        else:
            HLink=HLink.replace("/", "§")    

        command_parts = command.split("GET/entities/")
        command=command_parts[1]
        if(command[0]=="?"):
            command=command[1:]
        command_parts = command.split("&")
       
        
        for current in command_parts:
            current=current.split("=", 1)
            print(current[0])
            if(current[0]=="id"):
                print("id detected")
                entity_id_flag=True
                id=current[1]
            elif(current[0]=="idPattern"):
                entity_id_pattern_flag=True
                print("id pattern detected")

            elif(current[0]=="type"):
                entity_type_flag=True
                typee_multi=current[1].split(',')
                print("type detected")

            elif(current[0]=="time"):
                timee=current[1]
                print("time detected")  

            elif(current[0]=="limit"):
                limit=int(current[1])
                print("pagination limit detected")        

            elif(current[0]=="attrs"):
                entity_attrs_flag=True
                attrs=current[1].split(',')
                print("attrs detected")

            elif(current[0]=="q"):
                entity_query_flag=True 
                query=current[1]
                print("query detected")

            #Geoqueries    
            elif(current[0]=="geoproperty"):
                
                geoproperty=current[1]
                print("geoproperty detected") 
            elif (current[0]=="geometry"):
                geometry=current[1]
                print("geometry detected")
                geovar_count+=1
            elif (current[0]=="georel"):
                georel=current[1]
                print("geometry detected")
                geovar_count+=1
            elif (current[0]=="coordinates"):
                coordinates=current[1]
                print("coordinates detected")
                geovar_count+=1                  
            elif(current[0]=="area"):
                area=current[1].split(',')    
            else:
                print("Query not recognised")
                return    
            
        if(geovar_count!=0 and geovar_count!=3):
            print("Incomplete geoquery!")
            return                                
            
        if(area==[]):
            area.append('+')

        if(entity_type_flag==False):
            typee_multi=[1]
        if(entity_id_flag==False):
            id="#"      
        for typee in typee_multi:
            messages_for_context=[]
            check_top=[]
            if(typee==1):
                typee="#"

            for z in area:
                if(singleidadvertisement==False):
                                   
                    check_topic2="provider/+/+/"+z+'/'+HLink+'/'+typee
                else:
                    if(typee=="#"):
                        typee="+"
                                  
                    check_topic2="provider/+/+/"+z+'/'+HLink+'/'+typee+'/'+id       
                        
                check_top.append(check_topic2)
            if(Forwarding==1):    
                messages_for_context=GET(broker,port,check_top,0.1,1)
            if(typee=="#"):
                typee="+"
            context_providers_addresses=[]
            context_providers_ports=[]
            context_providers_areas=[]
            context_providers_full=[]

            if (Forwarding==1):    
                for messg in messages_for_context:
                    initial_topic=(messg.topic).split('/')
                    
                    if (initial_topic[1]+initial_topic[2]+initial_topic[3]) in context_providers_full:
                        continue
                    

                    context_providers_addresses.append(initial_topic[1])
                    context_providers_ports.append(initial_topic[2])
                    context_providers_areas.append(initial_topic[3])
                    context_providers_full.append(str(initial_topic[1]+initial_topic[2]+initial_topic[3]))
                    
                    top=initial_topic[3]
                    if attrs!='#':
                        
                        for i in attrs: 
                                top=initial_topic[3]+'/entities/'+HLink+'/'+typee+'/+/'+id+'/'+i
                                topic.append(top)

                    else:
                        top=initial_topic[3]+'/entities/'+HLink+'/'+typee+'/+/'+id+'/#'
                        topic.append(top) 
                    messages=GET(initial_topic[1],int(initial_topic[2]),topic,0.5,1,limit)                              
                    if(messages!=[]):
                        recreate_multiple_entities(messages,query,attrs,timee=timee,limit=limit,georel=georel,geometry=geometry,coordinates=coordinates,geoproperty=geoproperty,context_given=HLink)            
            else:   
                    print("Forwarding left by default for now")

                
    elif(re.search("entityOperations/delete",command)):
        if file=='':
            usage()
            sys.exit(2)
        print("\n ngsild Batch delete entity command detected \n")
        if (HLink==''):
            HLink='+'
            context_flag=False
        else:
            HLink=HLink.replace("/", "§")
        with open(file) as json_file:
            try:
                json_obj=json.load(json_file)
                for id in json_obj:
                    detected=1
                    typee='+'
                    top=my_area+'/entities/'+HLink+'/+/+/'+id+'/#' 
                    if (check_existence(broker,port,top)==False):
                        print("Entity with this id doesn't exist, no need for deletion")
                        detected=0
                    if detected==1:
                        T1 = threading.Thread(target = clear_retained,args=(broker,port,top))
                        T1.start()
                        print("Trying to delete content")
                        T1.join()
                        print("Content deletion complete")
                        tp=exists_topic.split("/")[-4]
                        top_check_for_advert=my_area+'/entities/'+HLink+'/'+tp+'/#'
                        
                        #remove from context source tables
                        if(singleidadvertisement==False):
                            if (check_existence(broker,port,top_check_for_advert )==False):
                                special_context_provider_broadcast= 'provider/' + broker + '/' +str(port) + '/' + my_area + '/' + HLink + '/+'
                                T1 = threading.Thread(target = clear_retained,args=(broker,port,special_context_provider_broadcast))
                                T1.start()
                                print("Trying to delete content")
                                T1.join()
                                print("Content deletion complete") 
                        else:
                            special_context_provider_broadcast= 'provider/' + broker + '/' +str(port) + '/'+my_area+'/' + context + '/' +typee +'/'+id
                            T1 = threading.Thread(target = clear_retained,args=(broker,port,special_context_provider_broadcast))
                            T1.start()
                            print("Trying to delete content")
                            T1.join()
                            print("Content deletion complete")              
            except:
                print("Couldn't completely parse the input file, are you sure it is valid json?")
                sys.exit(2)    
    elif(re.search("entityOperations/create",command)):
        advertisement_exists={}
        print("creating new instance")
        client = mqtt.Client(clean_session=True) #create new instance
        client.on_message=on_message #attach function to callback
        print("connecting to broker")
        client.connect(broker,port) #connect to broker
        if file=='':
            usage()
            sys.exit(2)
        print("\n ngsild Batch create entity command detected \n")
        with open(file) as json_file:
            try:
                json_obj=json.load(json_file)
                 
                for data in json_obj:
                    #cache type to avoid checking from broker 
                    if singleidadvertisement==False:
                        typee="error_no_type"
                        if 'type' in data:
                            typee=str(data['type'])

                        if  typee in advertisement_exists.keys():
                            post_entity(data,my_area,broker,port,qos,my_loc,1,client)
                        else:
                            post_entity(data,my_area,broker,port,qos,my_loc,0,client)
                            advertisement_exists.setdefault(typee,[])
                    else:        
                        post_entity(data,my_area,broker,port,qos,my_loc,0,client)
            except:
                print("Couldn't completely parse the input file, are you sure it is valid json?")
                sys.exit(2)    
        #client.loop_stop()        
    elif(re.search("entityOperations/update",command)):
        print("creating new instance")
        client = mqtt.Client(clean_session=True) #create new instance
        client.on_message=on_message #attach function to callback
        print("connecting to broker")
        client.connect(broker,port) #connect to broker
        if file=='':
            usage()
            sys.exit(2)
        print("\n ngsild Batch update entity command detected \n")
        with open(file) as json_file:
            try:
                json_obj=json.load(json_file)
                for data in json_obj:
                    post_entity(data,my_area,broker,port,qos,my_loc,1,client)
            except:
                print("Couldn't completely parse the input file, are you sure it is valid json?")
                sys.exit(2)        
    elif(re.search("entityOperations/upsert",command)):
        print("creating new instance")
        client = mqtt.Client(clean_session=True) #create new instance
        client.on_message=on_message #attach function to callback
        print("connecting to broker")
        client.connect(broker,port) #connect to broker
        if file=='':
            usage()
            sys.exit(2)
        print("\n ngsild Batch upsert entity command detected \n")
        with open(file) as json_file:
            try:
                json_obj=json.load(json_file)
                for data in json_obj:
                    #cache type to avoid checking from broker 
                    post_entity(data,my_area,broker,port,qos,my_loc,1,client)
            except:
                print("Couldn't completely parse the input file, are you sure it is valid json?")
                sys.exit(2)  
                              
           

                                                                   
#Check if the script is being run directly
#Retrieve command-line arguments passed to the script
#Call the main function with the command-line arguments

if __name__ == "__main__":
   main(sys.argv[1:])
            
