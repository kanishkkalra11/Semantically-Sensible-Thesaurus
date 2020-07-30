from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pickle
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet

# from verbreplacements_tensorapi import verb_replacements
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
import textacy
import spacy
nlp = spacy.load('en_core_web_sm')
import numpy as np
import math
from scipy import spatial
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to('cuda')
Verb_list = [(12822, 'is'), (10579, "'s"), (10044, 'wa'), (5115, 'had'), (4551, 'have'), (4043, 'are'), (3757, 'got'), (2689, 'want'), (2454, 'were'), (2135, 'ha'), (1402, 'need'), (1398, 'took'), (1396, 'told'), (1377, 'put'), (1285, 'get'), (1255, 'gave'), (1157, 'know'), (1146, 'give'), (1130, 'take'), (988, 'used'), (986, 'do'), (885, 'wanted'), (843, "'ll have"), (824, "'s got"), (818, 'seemed'), (818, 'like'), (794, 've got'), (783, 'said'), (765, 'gon'), (723, "'re"), (717, 'made'), (689, 'see'), (684, 'would be'), (684, 'is not'), (673, 'seems'), (652, 'tell'), (651, 'saw'), (629, 'make'), (619, 'going'), (615, "'s not"), (592, 'did'), (571, 'asked'), (557, 'will be'), (522, 'had been'), (514, 'include'), (509, 'left'), (508, 'be'), (496, 'mean'), (442, 'found'), (442, "do n't want"), (431, 'kept'), (429, 'tried'), (413, 'began'), (405, "is n't"), (401, 'keep'), (378, 'called'), (378, "'re gon"), (376, 'bought'), (374, 'call'), (372, 'knew'), (366, "was n't"), (348, 'say'), (347, 'seem'), (335, 'will have'), (334, 'has been'), (331, 'done'), (329, 'ought'), (323, "'ll give"), (321, 'show'), (317, 'use'), (317, "'s gon"), (316, 'brought'), (307, 'started'), (302, 'would have'), (298, 'can have'), (284, 'showed'), (281, 'turned'), (279, 'lost'), (278, "have n't got"), (270, 'must be'), (269, 'was not'), (268, 'love'), (267, 'seen'), (267, 'may be'), (266, "'ll get"), (265, 'held'), (256, 'won'), (251, 'was going'), (248, 'shook'), (248, "do n't have"), (246, 'included'), (245, "do n't know"), (243, 'heard'), (240, 'leave'), (236, "'"), (235, "'re going"), (232, "do n't like"), (231, 'sent'), (231, "'s going"), (230, 'needed'), (228, 'find'), (224, "'d like"), (223, 'set'), (217, 'felt'), (217, 'been'), (215, 'hit'), (213, 'liked'), (213, 'became'), (212, 'opened'), (212, 'includes'), (210, "'ll be"), (209, 'doe'), (208, 'was gon'), (208, 'remember'), (207, 'provides'), (206, 'reached'), (203, 'can do'), (202, 'spent'), (201, 'picked'), (201, 'paid'), (200, 'could be'), (199, 'can get'), (197, "'ll do"), (195, 'might be'), (195, 'involves'), (190, 'start'), (190, 'received'), (188, 'offer'), (188, 'decided'), (186, 'ask'), (184, 'tend'), (183, 'met'), (181, 'think'), (179, 'should be'), (178, 'play'), (176, 'have been'), (174, "'ll tell"), (172, "'d have"), (170, 'read'), (168, 'go'), (166, 'doing'), (165, 'try'), (165, 'had taken'), (165, "did n't have"), (165, 'caught'), (165, 'bring'), (164, 'refused'), (163, 'wish'), (161, 'cost'), (161, 'continued'), (156, 'would like'), (156, 'could see'), (154, 'requires'), (153, 'raised'), (153, 'beat'), (153, 'appears'), (152, 'meant'), (151, 'can be'), (150, 'followed'), (149, 'stopped'), (149, 'produce'), (149, 'appeared'), (147, 'pulled'), (146, 'pushed'), (145, 'managed'), (144, 'provide'), (143, 'watched'), (143, 'turn'), (143, 'offered'), (143, "do n't need"), (141, "are n't"), (140, "ca n't do"), (140, 'buy'), (139, 'had had'), (139, 'face'), (138, 'threw'), (138, 'led'), (137, "'ll take"), (134, 'provided'), (134, 'hold'), (134, 'help'), (134, 'cut'), (134, 'can see'), (134, 'being'), (133, 'us'), (133, 'joined'), (133, 'am'), (131, "did n't want"), (130, 'produced'), (129, 'contains'), (128, 'wrote'), (128, 'wore'), (128, 'pay'), (127, 'played'), (126, 'finished'), (125, 'had given'), (125, 'eat'), (124, 'helped'), (124, 'could do'), (122, 'killed'), (122, 'have got'), (121, 'may have'), (121, 'is going'), (121, 'failed'), (121, 'enjoyed'), (119, 'remains'), (118, 'thought'), (117, 'having'), (116, 'missed'), (115, 'can take'), (115, "'ll put"), (113, 'are not'), (113, 'are going'), (113, 'added'), (112, 'loved'), (111, 'will take'), (108, 'represents'), (108, 'hear'), (107, 'passed'), (107, 'hate'), (107, 'carried'), (106, 'was wearing'), (106, 'sold'), (105, 'write'), (105, 'remembered'), (105, 'had seen'), (105, 'had made'), (105, 'considered'), (104, 'was trying'), (104, 'ran'), (104, 'had told'), (104, 'drew'), (104, "did n't know"), (104, 'carry'), (103, "were n't"), (103, 'has had'), (103, 'becomes'), (103, "'ll see"), (102, 'win'), (102, 'look'), (102, 'appear'), (101, 'have had'), (101, "'d be"), (100, 'will give'), (100, 'owe'), (100, 'forced'), (99, 'would give'), (98, 'trying'), (98, 'require'), (97, 'given'), (97, 'choose'), (96, 'closed'), (95, 'handed'), (95, 'broke'), (94, 'might have'), (94, 'begin'), (93, 'stop'), (93, 'should have'), (93, 'leaf'), (93, 'brings'), (93, "'s had"), (92, 'affect'), (91, 'feel'), (90, 'went'), (90, 'was doing'), (90, 'hope'), (90, "have n't"), (90, 'form'), (89, 'would have been'), (89, 'describes'), (89, 'could have'), (89, 'caused'), (87, 'supposed'), (87, "'re not"), (86, 'can use'), (85, 'cover'), (84, 'understand'), (84, 'proved'), (84, 'had left'), (83, 'taking'), (83, "'s been"), (82, 'looked'), (81, 'pick'), (81, 'involve'), (81, 'ignored'), (81, 'agreed'), (80, 've had'), (80, 'contained'), (80, 'changed'), (79, 'sell'), (79, 'ordered'), (79, 'open'), (79, 'must have been'), (79, "'re doing"), (77, 'reported'), (77, 'not gon'), (77, 'must have'), (77, 'follow'), (77, 'enables'), (77, 'came'), (76, 'suffered'), (76, 'struck'), (76, 'shot'), (76, 'reminded'), (76, 'is called'), (75, 'noticed'), (75, 'could get'), (75, "'d had"), (74, 'wear'), (74, 'run'), (73, 'would take'), (73, 'were going'), (73, 'scored'), (73, 'rang'), (73, 'present'), (73, 'poured'), (73, 'dropped'), (73, 'change'), (72, 'lead'), (72, 'lay'), (72, 'hated'), (72, "did n't like"), (71, 'were not'), (71, 'was given'), (71, 'to be'), (71, 'spend'), (71, 'lifted'), (71, "'s doing"), (70, 'taken'), (70, 'entered'), (70, 'enabled'), (69, "wo n't be"), (69, 'seek'), (69, 'plan'), (69, 'moved'), (69, "do n't get"), (69, 'can put'), (69, 'believe'), (68, 'watch'), (68, "have n't seen"), (68, 'become'), (67, 'prefer'), (67, 'enjoy'), (67, 'described'), (67, 'consider'), (67, 'accepted'), (66, 'is expected'), (66, 'involved'), (66, 'grabbed'), (66, 'can give'), (66, 'attempt'), (66, 'aim'), (65, 'to do'), (65, 'tends'), (65, 'share'), (65, 'has got'), (65, 'contain'), (64, "'s called"), (63, 'was supposed'), (63, 'meet'), (63, 'has taken'), (63, 'continue'), (63, 'constitutes'), (63, "'d got"), (62, 'throw'), (62, 'raise'), (62, 'has become'), (62, "do n't"), (62, 'chose'), (62, 'can make'), (62, "ca n't get"), (62, "ai n't got"), (61, 'to take'), (61, 'reach'), (61, 'can be used'), (61, 'add'), (60, 'visited'), (60, 'urged'), (60, 'sought'), (60, 'signed'), (60, "do n't mind"), (60, 'could hear'), (60, 'can tell'), (60, "ca n't have"), (60, 'admitted'), (59, 'promised'), (59, 'had done'), (59, 'ate'), (59, "'ll leave"), (58, 'move'), (58, 'miss'), (58, "did n't see"), (57, 've put'), (57, 'suggested'), (57, 'required'), (57, 'refuse'), (57, 'owns'), (57, 'making'), (57, 'is said'), (57, "has n't got"), (57, 'expected'), (57, 'could take'), (56, 'support'), (56, 'stood'), (56, 'preferred'), (56, 'let'), (56, 'expect'), (56, 'discus'), (56, 'can find'), (55, 'wished'), (55, 'will make'), (55, 'represented'), (55, 'reflects'), (55, 'reflect'), (55, 'presented'), (55, 'lose'), (55, 'introduced'), (55, 'follows'), (55, "ca n't see"), (55, 'announced'), (54, 'to have'), (54, 'should take'), (54, 'save'), (54, 'pull'), (54, 'knocked'), (54, 'drove'), (54, 'come'), (54, 'claim'), (54, "ca n't afford"), (53, 'will do'), (53, 'warned'), (53, 'had lost'), (53, "do n't do"), (53, "did n't get"), (53, 'developed'), (53, 'could make'), (53, "'re not gon"), (52, 'was beginning'), (52, 'threatened'), (52, 'shall have'), (52, 'intended'), (52, 'had spent'), (52, 'had brought'), (52, 'getting'), (52, "'re supposed"), (51, 'thank'), (51, 'suggests'), (51, 'send'), (51, 'recognised'), (51, 'has made'), (51, 'had become'), (51, "did n't tell"), (51, 'continues'), (51, 'cause'), (51, "'s done"), (51, "'ll buy"), (50, 'work'), (50, 'were given'), (50, 'treat'), (50, 'owed'), (50, 'never had'), (50, 'lack'), (50, 'invited'), (50, 'do not have'), (50, "do n't wan"), (50, "did n't say"), (50, 'crossed'), (49, 'taught'), (49, 'pressed'), (49, 'phoned'), (49, 'increase'), (49, 'have taken'), (49, 'have seen'), (49, 'has given'), (49, 'created'), (49, 'could have been'), (49, "'re having"), (48, 'represent'), (48, 'persuaded'), (48, 'is used'), (48, "have n't had"), (48, 'doubt'), (48, "does n't want"), (48, "did n't do"), (48, 'creates'), (47, 'would need'), (47, 'would do'), (47, "wo n't have"), (47, 'pas'), (47, 'laid'), (47, 'fails'), (47, 'did not want'), (47, 'built'), (47, 'attracted'), (47, 'answered'), (46, 'would seem'), (46, 'would make'), (46, 'will get'), (46, 'was having'), (46, 'remained'), (46, 'marked'), (46, 'gained'), (46, 'faced'), (46, 'draw'), (46, 'believed'), (45, 'worked'), (45, 'was forced'), (45, 'tended'), (45, 'telling'), (45, 'place'), (45, "'ll find"), (44, 'kill'), (44, 'indicates'), (44, 'have made'), (44, 'had received'), (44, 'formed'), (44, 'expressed'), (44, 'drive'), (44, 'demanded'), (44, 'completed'), (44, 'checked'), (43, 'will continue'), (43, 'were gon'), (43, 'touched'), (43, 'regard'), (43, 'join'), (43, 'denied'), (43, 'catch'), (43, "'ll need"), (42, 'will need'), (42, 'using'), (42, 'shared'), (42, 'recorded'), (42, 'recognized'), (42, 'implies'), (42, 'filled'), (42, 'explained'), (42, 'dragged'), (42, "do n't see"), (42, 'did not have'), (42, 'describe'), (42, 'comprises'), (42, 'claimed'), (42, 'are expected'), (41, 'was taking'), (41, 'to give'), (41, 'stole'), (41, 'shouted'), (41, 'proposed'), (41, 'known'), (41, 'covered'), (41, "ca n't remember"), (40, 'was found'), (40, 'touch'), (40, 'to get'), (40, 'revealed'), (40, 'rejected'), (40, 'manage'), (40, 'have done'), (40, 'had asked'), (40, 'forgot'), (40, 'attempted'), (40, 'allowed'), (40, "'re trying"), (39, 'will enable'), (39, 'was called'), (39, 'serve'), (39, 'regarded'), (39, 'mentioned'), (39, "does n't have"), (39, 'break'), (39, 'accept'), (39, "'ll make"), (39, "'d get"), (38, 'served'), (38, 'placed'), (38, 'owes'), (38, 'has won'), (38, 'had known'), (38, 'had got'), (38, 'express'), (38, 'ended'), (38, "did n't seem"), (38, 'can buy'), (38, 'blame'), (38, 'approached'), (38, 'achieved'), (38, 'accused'), (38, "'s supposed"), (38, "'re getting"), (37, 've been'), (37, 'serf'), (37, 'receive'), (37, 'kissed'), (37, 'illustrates'), (37, 'fail'), (37, 'enjoys'), (37, 'employ'), (37, 'discovered'), (37, "did n't give"), (37, 'constitute'), (37, 'are trying'), (37, "ai n't"), (37, "'ll show"), (36, "would n't have"), (36, 'would have had'), (36, 'will tell'), (36, 'was made'), (36, 'wan'), (36, 'supported'), (36, 'saved'), (36, 'may have been'), (36, 'mark'), (36, 'learned'), (36, 'is thought'), (36, 'invite'), (36, 'had tried'), (36, 'giving'), (36, 'did have'), (36, 'could put'), (36, 'can help'), (36, 'asks'), (36, "'s taking"), (36, "'d give"), (35, "wo n't do"), (35, 'was telling'), (35, 'teach'), (35, 'putting'), (35, 'might get'), (35, 'learnt'), (35, 'lacked'), (35, 'indicate'), (35, 'have told'), (35, 'has seen'), (35, 'had used'), (35, 'had heard'), (35, 'encouraged'), (35, 'denies'), (35, 'concern'), (35, 'can say'), (35, 'can'), (35, 'are given'), (35, "'ll bring"), (34, "wo n't get"), (34, 'will require'), (34, 'will help'), (34, 'welcomed'), (34, 'understood'), (34, 'sat'), (34, 'rubbed'), (34, 'report'), (34, 'learn'), (34, 'intend'), (34, 'hurt'), (34, 'has put'), (34, 'fear'), (34, 'do have'), (34, 'dismissed'), (34, "did n't"), (34, 'bear'), (34, 'backed'), (34, "'s trying"), (34, "'ll keep"), (33, 'studied'), (33, 'spell'), (33, 'sends'), (33, 'promise'), (33, 'loses'), (33, 'launched'), (33, 'had put'), (33, 'discussed'), (33, 'collected'), (33, 'can afford'), (33, "'s not gon"), (33, "'re not going"), (32, '—'), (32, 'will include'), (32, 'were having'), (32, 'were found'), (32, 'was giving'), (32, 'visit'), (32, 'swung'), (32, 'reduce'), (32, 'might have been'), (32, 'increased'), (32, 'had started'), (32, 'had found'), (32, 'encourage'), (32, 'deserves'), (32, 'dare'), (32, 'collect'), (32, 'attended'), (32, 'are gon'), (32, "'d do"), (31, "would n't want"), (31, 'would get'), (31, 'would bring'), (31, 'wiped'), (31, 'will find'), (31, 'waved'), (31, 'was saying'), (31, 'reduced'), (31, 'lit'), (31, "have n't done"), (31, 'have lost'), (31, 'has done'), (31, 'has been appointed'), (31, 'greeted'), (31, 'earned'), (31, "do n't give"), (31, 'decide'), (31, 'could tell'), (31, "could n't get"), (31, 'could give'), (31, 'can keep'), (31, 'arranged'), (31, 'are doing'), (31, 'approach'), (30, "would n't like"), (30, "would n't be"), (30, 'washed'), (30, 'was making'), (30, 'voted'), (30, 'suggest'), (30, 'stress'), (30, 'snapped'), (30, 'retained'), (30, 'removed'), (30, 'reflected'), (30, 'planned'), (30, 'owned'), (30, 'married'), (30, 'is required'), (30, 'have used'), (30, 'had turned'), (30, 'had said'), (30, 'had reached'), (30, 'had bought'), (30, 'had begun'), (30, 'fill'), (30, 'experienced'), (30, 'enable'), (30, "does n't like"), (30, 'could use'), (30, 'assured'), (30, 'advised'), (30, "'s getting"), (30, "'d seen"), (29, 'yield'), (29, 've told'), (29, 'tore'), (29, 'surprised'), (29, 'stuck'), (29, 'stretched'), (29, 'spoke'), (29, 'shut'), (29, 'repeat'), (29, 'push'), (29, 'prompted'), (29, 'prepared'), (29, 'may not be'), (29, 'happened'), (29, 'had seemed'), (29, 'had kept'), (29, 'fit'), (29, 'erm'), (29, 'create'), (29, "'s not going"), (28, 'treated'), (28, 'switched'), (28, 'slipped'), (28, 'shrugged'), (28, 'reject'), (28, 'pointed'), (28, 'obtained'), (28, 'kicked'), (28, 'is made'), (28, 'have given'), (28, 'have found'), (28, 'has told'), (28, 'has lost'), (28, 'happens'), (28, 'happen'), (28, 'had met'), (28, 'examined'), (28, 'drank'), (28, 'does not have'), (28, 'could find'), (28, 'can write'), (28, "ca n't tell"), (28, 'bother'), (28, 'apply'), (28, "'re making"), (28, "'ll"), (27, 'would provide'), (27, "would n't do"), (27, 'will cost'), (27, 'were doing'), (27, 'to make'), (27, 'round'), (27, 'issued'), (27, 'is set'), (27, 'intends'), (27, 'imagine'), (27, 'have tried'), (27, 'have received'), (27, 'has shown'), (27, 'drop'), (27, "did n't need"), (27, 'define'), (27, 'defeated'), (27, 'declined'), (27, 'blew'), (27, 'affected'), (27, "'re not having"), (26, 'would help'), (26, 'will see'), (26, 'will receive'), (26, 'will meet'), (26, 'spread'), (26, 'shown'), (26, 'reveals'), (26, 'returned'), (26, 'replaced'), (26, 'recalled'), (26, 'posse'), (26, 'permit'), (26, 'must take'), (26, 'may take'), (26, 'leaving'), (26, 'identified'), (26, 'has brought'), (26, 'had decided'), (26, 'finish'), (26, 'fancy'), (26, 'explain'), (26, 'expects'), (26, 'end'), (26, "does n't seem"), (26, 'does have'), (26, "do n't understand"), (26, "do n't think"), (26, 'demand'), (26, 'cleared'), (26, "ca n't be"), (26, 'am going'), (26, 'allows'), (26, "'s having"), (26, "'ll ask"), (26, "'d done"), (25, 'would keep'), (25, 'would have liked'), (25, 'will tend'), (25, 'wash'), (25, "was n't going"), (25, 've taken'), (25, 'switch'), (25, 'suit'), (25, 'speak'), (25, 'sounded'), (25, 'sipped'), (25, 'rolled'), (25, 'recognise'), (25, 'occupied'), (25, 'lowered'), (25, 'is trying'), (25, 'is gon'), (25, 'have become'), (25, "has n't"), (25, 'has left'), (25, 'examines'), (25, 'did not like'), (25, 'could say'), (25, 'could feel'), (25, 'could afford'), (25, 'can hear'), (25, 'bore'), (25, 'bet'), (25, "'d put"), (24, 'would put'), (24, 'would prefer'), (24, 'would have done'), (24, 'will provide'), (24, 'waited'), (24, 'suited'), (24, 'strike'), (24, 'sound'), (24, 'should give'), (24, 'should get'), (24, 'propose'), (24, 'obtain'), (24, 'mention'), (24, 'may seem'), (24, 'is believed'), (24, 'indicated'), (24, 'highlight'), (24, 'have shown'), (24, 'handle'), (24, 'had sent'), (24, 'had managed'), (24, 'feature'), (24, 'drink'), (24, 'disliked'), (24, "did n't take"), (24, 'demonstrated'), (24, 'could bring'), (24, 'control'), (24, 'climbed'), (24, "ca n't find"), (24, 'are having'), (24, 'afford'), (24, 'adopted'), (23, "would n't mind"), (23, 'would have given'), (23, 'worry'), (23, 'will bring'), (23, 'was not going'), (23, 've never seen'), (23, 've lost'), (23, 've found'), (23, 'unveiled'), (23, 'underwent'), (23, 'slammed'), (23, 'remove'), (23, 'point'), (23, 'perform'), (23, 'nodded'), (23, 'lift'), (23, 'has spent'), (23, 'had been given'), (23, 'gathered'), (23, 'forget'), (23, 'fed'), (23, 'exhibit'), (23, 'equal'), (23, 'did not seem'), (23, 'declared'), (23, "could n't believe"), (23, 'cast'), (23, 'blocked'), (23, "'re giving"), (23, "'ll try"), (23, "'d love"), (22, 'would want'), (22, 'were taking'), (22, 'watching'), (22, 'was holding'), (22, 've heard'), (22, 've done'), (22, 'suffer'), (22, 'ring'), (22, 'replied'), (22, 'reminds'), (22, 'reduces'), (22, 'receives'), (22, 'organised'), (22, 'opposed'), (22, 'must do'), (22, 'might like'), (22, 'limit'), (22, 'lent'), (22, 'is taking'), (22, 'is bound'), (22, 'hired'), (22, 'have tended'), (22, 'has led'), (22, "had n't got"), (22, 'glimpsed'), (22, 'flung'), (22, 'favour'), (22, 'encourages'), (22, 'eats'), (22, "does n't make"), (22, "does n't know"), (22, "do n't seem"), (22, 'destroyed'), (22, 'demonstrates'), (22, 'count'), (22, "could n't see"), (22, 'ceased'), (22, 'can start'), (22, 'can provide'), (22, "ca n't help"), (22, 'borrow'), (22, 'addressed'), (22, "'s given"), (22, "'d want"), (21, 'will use'), (21, 'will keep'), (21, 'welcome'), (21, 'was told'), (21, 'was carrying'), (21, 'view'), (21, 'to put'), (21, 'stand'), (21, 'spends'), (21, 'sensed'), (21, 'sang'), (21, 'remain'), (21, 'notice')]
finalnounslist = [(22653, 'i'), (18488, 'you'), (17639, 'it'), (16419, 'he'), (10538, 'there'), (9515, 'they'), (9504, 'we'), (9105, 'she'), (8974, 'it'), (7059, 'that'), (4320, 'who'), (4289, 'which'), (3645, 'one'), (3137, 'you'), (2992, 'him'), (2826, 'them'), (2708, 'that'), (2685, 'this'), (2377, 'me'), (1862, 'her'), (1617, '’'), (1257, 'something'), (1205, 'one'), (1037, 'to'), (1002, 'thing'), (920, 'way'), (899, 'time'), (860, 'nothing'), (829, 'lot'), (826, 'this'), (801, '’'), (770, 'people'), (697, 'anything'), (694, '—'), (687, 'm'), (674, 'u'), (671, 'what'), (656, 'place'), (582, 'a'), (567, 'problem'), (555, 'part'), (538, 'more'), (520, 'money'), (519, 'all'), (516, 'man'), (513, 'head'), (508, 'for'), (506, 'hand'), (476, 'people'), (472, 'bit'), (456, 'some'), (439, 'number'), (432, 'these'), (430, 'two'), (426, 'idea'), (419, 'point'), (410, 'look'), (409, 'me'), (407, 'question'), (405, 'himself'), (398, 'man'), (392, 'job'), (373, 'word'), (371, 'much'), (359, 'same'), (349, 'chance'), (348, 'name'), (344, 'case'), (338, 'woman'), (335, 'house'), (330, 'woman'), (323, 'the'), (323, 'day'), (321, 'work'), (313, 'child'), (312, 'car'), (304, "'s"), (302, 'all'), (300, 'm'), (299, 'sort'), (297, 'form'), (294, 'eye'), (289, 'i'), (273, 'reason'), (270, 'pound'), (269, 'thing'), (264, 'door'), (264, 'difference'), (263, 'some'), (261, 'everything'), (257, 'themselves'), (256, 'those'), (256, 'hundred'), (255, 'someone'), (255, 'need'), (255, 'kind'), (254, 'life'), (254, 'any'), (251, 'girl'), (249, 'room'), (247, 'face'), (245, 'herself'), (243, 'adam'), (240, 'government'), (239, 'three'), (238, 'erm'), (237, 'him'), (236, 'child'), (236, 'right'), (234, 'year'), (231, 'letter'), (231, 'effect'), (230, 'five'), (225, 'matter'), (217, 'evidence'), (216, 'four'), (215, 'go'), (211, 'change'), (209, 'game'), (207, 'group'), (207, 'view'), (205, 'police'), (203, 'company'), (202, 'corbett'), (199, 'use'), (198, 'story'), (196, '…'), (196, '‘'), (196, 'book'), (195, 'them'), (194, 'picture'), (194, 'friend'), (190, 'other'), (189, 'interest'), (187, 'u'), (187, 'boy'), (186, 'men'), (181, 'sense'), (180, 'little'), (177, 'thousand'), (177, 'attention'), (176, 'sign'), (176, 'body'), (174, 'feeling'), (173, 'mum'), (173, 'itself'), (172, 'drink'), (172, 'answer'), (170, 'hour'), (169, 'those'), (167, 'result'), (165, 'cup'), (164, 'they'), (164, 'piece'), (162, 'number'), (161, 'party'), (161, 'side'), (160, 'trouble'), (159, 'law'), (159, 'girl'), (159, 'voice'), (158, 'deal'), (156, 'problem'), (154, 'mind'), (153, 'couple'), (152, 'many'), (151, 'person'), (151, 'information'), (150, 'others'), (150, 'father'), (150, 'minute'), (149, 'end'), (149, 'choice'), (149, 'arm'), (149, 'area'), (147, 'goal'), (146, 'mother'), (146, 'someone'), (146, 'power'), (146, 'example'), (145, '—'), (145, 'role'), (145, 'plenty'), (144, 'plan'), (143, 'anyone'), (142, 'her'), (140, 'somebody'), (140, 'opportunity'), (139, 'dad'), (139, 'going'), (139, 'fact'), (138, 'patient'), (136, 'step'), (136, 'men'), (136, 'business'), (135, 'study'), (135, 'value'), (134, 'group'), (133, 'line'), (133, 'feature'), (132, 'ruth'), (132, 'mine'), (132, 'member'), (131, 'tea'), (131, 'home'), (131, 'he'), (131, 'half'), (130, 'what'), (130, 'solution'), (130, 'round'), (130, 'enough'), (129, 'parent'), (129, 'possibility'), (129, 'element'), (129, 'difficulty'), (128, 'most'), (127, 'nobody'), (127, 'example'), (127, 'history'), (127, 'doubt'), (126, 'help'), (125, 'two'), (125, 'everyone'), (125, 'boy'), (125, 'record'), (124, 'person'), (124, 'stuff'), (124, 'party'), (124, 'argument'), (123, 'risk'), (123, 'relationship'), (123, 'paper'), (123, 'news'), (123, 'decision'), (122, 'school'), (122, 'glass'), (120, 'most'), (120, 'a'), (120, 'thought'), (119, 'these'), (119, 'love'), (119, 'law'), (118, 'system'), (117, 'work'), (117, 'six'), (116, 'water'), (115, 'name'), (114, 'result'), (114, 'function'), (113, 'seven'), (113, 'price'), (113, 'company'), (112, 'light'), (112, 'hair'), (112, 'foot'), (112, 'finger'), (111, 'issue'), (111, 'action'), (110, 'support'), (109, 'player'), (109, 'mother'), (109, 'food'), (109, 'card'), (108, 'mummy'), (108, 'we'), (108, 'mum'), (108, 'move'), (108, 'matrix'), (108, 'doing'), (107, 'wife'), (107, 'range'), (107, 'load'), (107, 'family'), (107, 'experience'), (107, 'best'), (106, 'set'), (106, 'death'), (105, 'history'), (105, 'edward'), (105, 'nine'), (105, 'mark'), (105, 'figure'), (105, '%'), (104, '‘'), (104, 'space'), (104, 'leg'), (104, 'advantage'), (103, 'house'), (103, 'type'), (103, 'service'), (103, 'see'), (102, 'subject'), (102, 'somebody'), (101, 'mistake'), (101, 'method'), (101, 'amount'), (100, 'share'), (100, 'myself'), (99, 'something'), (99, 'figure'), (99, 'week'), (99, 'good'), (99, 'approach'), (98, 'both'), (98, 'rule'), (97, 'doctor'), (97, 'rest'), (97, 'force'), (96, 'offence'), (96, 'account'), (95, 'way'), (95, 'message'), (95, 'many'), (95, 'bag'), (94, 'family'), (94, 'term'), (94, 'million'), (94, 'level'), (94, 'government'), (94, 'another'), (93, 'word'), (93, 'method'), (93, 'change'), (93, 'report'), (93, 'meeting'), (93, 'father'), (92, 'system'), (92, 'solution'), (92, 'increase'), (91, 'member'), (91, 'night'), (91, 'key'), (91, 'factor'), (91, 'eight'), (91, 'baby'), (90, 'prince'), (90, 'miranda'), (90, '%'), (90, 'team'), (90, 'responsibility'), (90, 'lady'), (90, 'do'), (90, 'claim'), (90, 'basis'), (90, 'attempt'), (89, 'everybody'), (89, 'ten'), (89, 'process'), (89, 'pair'), (89, 'note'), (89, 'ground'), (89, 'control'), (88, 'school'), (88, 'helen'), (88, 'response'), (88, 'fault'), (87, 'mark'), (87, 'court'), (87, 'aim'), (87, 'table'), (87, 'detail'), (86, 'act'), (86, 'yours'), (86, 'truth'), (86, 'start'), (86, 'situation'), (86, 'position'), (86, 'condition'), (86, 'age'), (85, 'time'), (85, 'none'), (85, 'moment'), (84, 'defendant'), (84, 'council'), (84, 'size'), (84, 'play'), (84, 'others'), (84, 'first'), (83, 'question'), (83, 'yourself'), (83, 'window'), (83, 'task'), (83, 'son'), (83, 'profit'), (83, 'office'), (83, 'none'), (83, 'measure'), (83, 'fear'), (83, 'charge'), (82, 'britain'), (82, 'wife'), (82, 'month'), (82, 'meaning'), (82, 'impression'), (82, 'fire'), (82, 'biscuit'), (82, 'ball'), (81, 'idea'), (81, 'friend'), (81, 'd'), (81, 'rise'), (81, 'open'), (81, 'hope'), (81, 'colour'), (80, 'road'), (80, 'quid'), (80, 'police'), (80, 'dream'), (79, 'team'), (79, 'phone'), (79, 'loss'), (79, 'lead'), (79, 'get'), (78, 'country'), (78, 'crime'), (78, 'concept'), (78, 'bill'), (78, 'back'), (78, 'aspect'), (77, 'record'), (77, 'point'), (77, 'player'), (77, 'god'), (77, 'world'), (77, 'row'), (77, 'box'), (77, 'benefit'), (76, 've'), (76, 'everything'), (76, 'each'), (76, 'sound'), (76, 'his'), (76, 'act'), (75, 'nothing'), (75, 'answer'), (75, 'she'), (75, 'hold'), (75, 'gun'), (75, 'coffee'), (75, 'anyone'), (74, 'eye'), (74, 'approach'), (74, 'victim'), (74, 'success'), (74, 'sentence'), (74, 'property'), (74, 'development'), (74, 'course'), (74, 'concern'), (73, 'anybody'), (73, 'order'), (73, 'class'), (73, 'chair'), (72, 'specie'), (72, 'face'), (72, 'product'), (72, 'option'), (72, 'damage'), (72, 'boat'), (72, 'access'), (71, 'side'), (71, 'rose'), (71, 'talk'), (71, 'horse'), (71, 'hat'), (71, 'dog'), (70, 'scott'), (70, 'car'), (70, 'study'), (70, 'second'), (70, 'lunch'), (70, 'fish'), (70, 'equation'), (70, 'danger'), (69, 'individual'), (69, 'daddy'), (69, 'b'), (69, 'rate'), (69, 'influence'), (69, 'clothes'), (68, 'report'), (68, 'part'), (68, 'lot'), (68, 'body'), (68, 'shot'), (68, 'mile'), (68, 'meal'), (68, 'dinner'), (68, 'character'), (67, 'hand'), (67, 'effect'), (67, 'case'), (67, 'turn'), (67, 'try'), (67, 'list'), (67, 'intention'), (67, 'degree'), (67, 'building'), (66, 'unionist'), (66, 'labour'), (66, 'clare'), (66, 'wall'), (66, 'source'), (66, 'sight'), (66, 'series'), (66, 'run'), (66, 'expression'), (66, 'clue'), (66, 'being'), (66, 'b'), (65, 'voice'), (65, 'experience'), (65, 'ellen'), (65, 'thinking'), (65, 'quarter'), (65, 'penny'), (65, 'model'), (65, 'match'), (65, 'confidence'), (65, 'bottle'), (64, 'speaker'), (64, 'reason'), (64, 'own'), (64, 'impact'), (64, 'hole'), (64, 'event'), (64, 'duty'), (64, 'country'), (64, 'contract'), (64, 'call'), (63, 'purpose'), (63, 'husband'), (63, 'element'), (63, 'area'), (63, 'zero'), (63, 'vote'), (63, 'top'), (63, 'seat'), (63, 'no'), (63, 'looking'), (63, 'heart'), (63, 'consequence'), (63, 'birthday'), (62, 'share'), (62, 'section'), (62, 'rufus'), (62, 'life'), (62, 'weight'), (62, 'war'), (62, 'title'), (62, 'relation'), (62, 'effort'), (62, 'doctor'), (62, 'brother'), (62, 'address'), (61, 'version'), (61, 'mean'), (61, 'lip'), (61, 'knowledge'), (61, 'getting'), (61, 'cost'), (60, 'teacher'), (60, 'student'), (60, 'reader'), (60, 'process'), (60, 'c'), (60, 'your'), (60, 'smile'), (60, 'sister'), (60, 'ring'), (60, 'policy'), (60, 'fifty'), (60, 'defence'), (60, 'dad'), (60, 'connection'), (60, 'coat'), (60, 'animal'), (60, 'agreement'), (59, 'son'), (59, 'plan'), (59, 'place'), (59, 'majority'), (59, 'form'), (59, 'factor'), (59, 'club'), (59, 'stage'), (59, 'kid'), (59, 'fun'), (59, 'field'), (59, 'er'), (59, 'discussion'), (59, 'affair'), (58, 'death'), (58, 'analyst'), (58, 'operation'), (58, 'importance'), (58, 'image'), (58, 'having'), (58, 'few'), (58, 'extension'), (58, 'constant'), (58, 'attitude'), (58, 'adam'), (57, 'matthew'), (57, 'variety'), (57, 'threat'), (57, 'taking'), (57, 'silence'), (57, 'potential'), (57, 'pattern'), (57, 'mummy'), (57, 'flower'), (57, 'explanation'), (57, 'daughter'), (57, 'care'), (57, 'bed'), (56, 'writer'), (56, 'proportion'), (56, 'issue'), (56, 'teeth'), (56, 'p'), (56, 'my'), (56, 'fruit'), (56, 'flat'), (56, 'accident'), (55, 'job'), (55, 'decision'), (55, 'book'), (55, 'authority'), (55, 'attempt'), (55, 'action'), (55, 'tree'), (55, 'test'), (55, 'talking'), (55, 'shop'), (55, 'principle'), (55, 'offer'), (55, 'film'), (55, 'english'), (55, 'dress'), (55, 'consideration'), (55, 'cigarette'), (55, 'behaviour'), (54, 'three'), (54, 'susan'), (54, 'sally'), (54, 'plant'), (54, 'up'), (54, 'theory'), (54, 'specie'), (54, 'put'), (54, 'performance'), (54, 'parent'), (54, 'mess'), (54, 'length'), (54, 'contact'), (54, 'attack'), (53, 'worker'), (53, 'argument'), (53, 'tape'), (53, 'street'), (53, 'statement'), (53, 'state'), (53, 'reading'), (53, 'quality'), (53, 'pain'), (53, 'owen'), (53, 'notion'), (53, 'link'), (53, 'column'), (53, 'appeal'), (53, 'activity'), (52, 'use'), (52, 'head'), (52, 'trick'), (52, 'structure'), (52, 'strength'), (52, 'shape'), (52, 'shame'), (52, 'pressure'), (52, 'milk'), (52, 'lad'), (52, 'hell'), (52, 'bread'), (51, 'rule'), (51, 'half'), (51, 'feature'), (51, 'driver'), (51, 'brother'), (51, 'bank'), (51, 'winner'), (51, 'twenty'), (51, 'ta'), (51, 'song'), (51, 'saying'), (51, 'return'), (51, 'music'), (51, 'husband'), (51, 'holiday'), (51, 'dead'), (50, 'value'), (50, 'tim'), (50, 'money'), (50, 'erm'), (50, 'equation'), (50, 'day'), (50, 'charlotte'), (50, 'treatment'), (50, 'teacher'), (50, 'show'), (50, 'race'), (50, 'programme'), (50, 'photograph'), (50, 'language'), (50, 'file'), (50, 'egg'), (50, 'copy'), (50, 'conversation'), (50, 'breath'), (50, 'bedroom'), (49, 'sartre'), (49, 'other'), (49, 'crime'), (49, 'couple'), (49, 'secret'), (49, 'progress'), (49, 'material'), (49, 'guy'), (49, 'demand'), (49, 'contribution'), (49, 'challenge'), (48, 'staff'), (48, 'judge'), (48, 'fact'), (48, 'candidate'), (48, 'tendency'), (48, 'skill'), (48, 'notice'), (48, 'nose'), (48, 'mouth'), (48, 'city'), (48, 'blood'), (48, 'authority'), (48, 'analysis'), (47, 'wave'), (47, 'user'), (47, 'thought'), (47, 'england'), (47, 'david'), (47, 'walk'), (47, 'visit'), (47, 'ticket'), (47, 'sum'), (47, 'score'), (47, 'scene'), (47, 'reaction'), (47, 'minister'), (47, 'memory'), (47, 'market'), (47, 'let'), (47, 'indication'), (47, 'improvement'), (47, 'hall'), (47, 'fortune'), (47, 'fight'), (46, 'term'), (46, 'officer'), (46, 'kind'), (46, 'being'), (46, 'taste'), (46, 'smell'), (46, 'period'), (46, 'noise'), (46, 'movement'), (46, 'making'), (46, 'individual'), (46, 'habit'), (46, 'extent'), (46, 'election'), (46, 'club'), (46, 'cat'), (45, 'view'), (45, 'need'), (45, 'king'), (45, 'is'), (45, 'force'), (45, 'difference'), (45, 'style'), (45, 'shoulder'), (45, 'proof'), (45, 'present'), (45, 'pity'), (45, 'leader'), (45, 'injury'), (45, 'helen'), (45, 'glance'), (45, 'excuse'), (45, 'advice'), (44, 'theory'), (44, 'peter'), (44, 'owen'), (44, 'minister'), (44, 'letter'), (44, 'latter'), (44, 'file'), (44, 'chapter'), (44, 'activity'), (44, 'win'), (44, 'train'), (44, 'track'), (44, 'target'), (44, 'take'), (44, 'say'), (44, 'reader'), (44, 'pen'), (44, 'page'), (44, 'mate'), (44, 'kiss'), (44, 'implication'), (44, 'exercise'), (44, 'director'), (44, 'definition'), (44, 'cover'), (44, 'court'), (44, 'church'), (44, 'chip'), (44, 'cake'), (43, 'plaintiff'), (43, 'operator'), (43, 'neither'), (43, 'martin'), (43, 'event'), (43, 'committee'), (43, 'video'), (43, 'unit'), (43, 'thirty'), (43, 'last'), (43, 'land'), (43, 'lack'), (43, 'knife'), (43, 'king'), (43, 'joke'), (43, 'investment'), (43, 'god'), (43, 'fall'), (43, 'eigenvalue'), (43, 'edward'), (43, 'cry'), (43, 'commitment'), (43, 'bath'), (43, 'anybody'), (43, 'air'), (43, 'ability'), (42, 'sister'), (42, 'matrix'), (42, 'kid'), (42, 'greg'), (42, 'development'), (42, 'bird'), (42, 'worker'), (42, 'twelve'), (42, 'touch'), (42, 'secretary'), (42, 'resource'), (42, 'priority'), (42, 'paul'), (42, 'nature'), (42, 'murder'), (42, 'kitchen'), (42, 'in'), (42, 'break'), (42, 'both'), (41, 'soldier'), (41, 'rest'), (41, 'relationship'), (41, 'paula'), (41, 'offence'), (41, 'model'), (41, 'lady'), (41, 'interest'), (41, 'georgina'), (41, 'forster'), (41, 'firm'), (41, 'buzz'), (41, 'warning'), (41, 'square'), (41, 'shoe'), (41, 'practice'), (41, 'playing'), (41, 'plant'), (41, 'penalty'), (41, 'patient'), (41, 'division'), (41, 'cut'), (41, 'cause'), (41, 'application'), (40, '…'), (40, 'step'), (40, 'room'), (40, 'performance'), (40, 'paul'), (40, 'lewis'), (40, 'foucault'), (40, 'business'), (40, 'author'), (40, 'animal'), (40, 'analysis'), (40, 'whole'), (40, 'visitor'), (40, 'variation'), (40, 'using'), (40, 'staff'), (40, 'spot'), (40, 'significance'), (40, 'sheet'), (40, 's'), (40, 'prospect'), (40, 'proportion'), (40, 'plate'), (40, 'object'), (40, 'manager'), (40, 'liability'), (40, 'energy'), (40, 'distance'), (40, 'data'), (40, 'cross'), (40, 'corner'), (40, 'comment'), (40, 'coming'), (40, 'combination'), (40, 'clear'), (40, 'cheque'), (40, 'centre'), (40, 'blow'), (40, 'appearance'), (39, 'wan'), (39, 'response'), (39, 'dog'), (39, 'do'), (39, 'daughter'), (39, 'wood'), (39, 'waste'), (39, 'volume'), (39, 'stick'), (39, 'sixty'), (39, 'sex'), (39, 'seeing'), (39, 'section'), (39, 'scheme'), (39, 'requirement'), (39, 'purpose'), (39, 'protection'), (39, 'phenomenon'), (39, 'path'), (39, 'map'), (39, 'le'), (39, 'desire'), (39, 'council'), (39, 'content'), (39, 'cash'), (39, 'capacity'), (39, 'bus'), (39, 'appointment'), (38, 'world'), (38, 'whoever'), (38, 'table'), (38, 'subject'), (38, 'leader'), (38, 'guy'), (38, 'first'), (38, 'wave'), (38, 'stone'), (38, 'reference'), (38, 'project'), (38, 'presence'), (38, 'post'), (38, 'pleasure'), (38, 'pint'), (38, 'opinion'), (38, 'mr'), (38, 'majority'), (38, 'lie'), (38, 'laugh'), (38, 'journey'), (38, 'bowl'), (38, 'bastard'), (38, 'bar'), (38, 'alternative'), (37, 'victim'), (37, 'scientist'), (37, 'project'), (37, 'much'), (37, 'level'), (37, 'language'), (37, 'jenny'), (37, 'concept'), (37, 'class'), (37, 'writing'), (37, 'tort'), (37, 'text')]
W = np.load("latentfactors.npy")
G = np.load("coretensor.npy")
def svocompositioncontextual(a,b,c):
    a_ind = -1
    b_ind = -1
    c_ind = -1
    for i in range(len(finalnounslist)):
        if (finalnounslist[i][1] == a):
            a_ind = i
        if (finalnounslist[i][1] == c):
            c_ind = i
        if (a_ind!=-1 and c_ind!=-1):
            break
    for i in range(len(Verb_list)):
        if (Verb_list[i][1] == b):
            b_ind = i
        if (b_ind!=-1):
            break

    if (a_ind==-1 or b_ind==-1 or c_ind==-1):
        return np.zeros(1) # if verb or subject or object not available in our training dataset

    s = W[a_ind]
    o = W[c_ind]
    Y = np.outer(s,o) # vector outer product

    Gv = G[b_ind]

    Z = np.multiply(Gv,Y) # Hadamard product
    return Z
def svocompositionnoncontextual(b):
    b_ind = -1
    for i in range(len(Verb_list)):
        if (Verb_list[i][1] == b):
            b_ind = i
        if (b_ind!=-1):
            break
    if (b_ind==-1):
        return np.zeros(1) # if verb not present in our training dataset

    Z = G[b_ind] # slice of core tensor
    return Z
def similarity(A,B):
    # a = A.flatten()
    # b = B.flatten()
    C = np.asmatrix(np.full((300,1),(float(1)/math.sqrt(300)))) # column vector to convert matrix into a vectorized and normalized representation
    a = np.asarray(np.dot(A,C)) # matrix multiplication to obtain column vector
    b = np.asarray(np.dot(B,C))
    return (1 - spatial.distance.cosine(a,b)) # cosine similarity
def verb_replacements(sub,verb,obj): #lemmatized
    similarities = []
    ans = []
    A = svocompositioncontextual(sub,verb,obj)
    if (A.shape == (np.zeros(1)).shape):
        return ans
    for i in range(1000):
        B = G[i]
        try:
            similarities.append((i, similarity(A,B)))
        except:
            continue
    similarities.sort(key = lambda x: x[1], reverse = True)
    for j in range(3):
        ans.append(Verb_list[similarities[j][0]][1])
    return ans

def pp_input(sent):
  	#text = "[CLS]" + sent + "[SEP]" + sent + "[CLS]"
  	tokenized_sent = tokenizer.tokenize(sent)
  	tokenized_sent_2 = tokenized_sent.copy()
  	verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
  	mask_indexes = []
  	tags = nltk.pos_tag(tokenized_sent)
  	#print(tokenized_sent)
  	for i in range(len(tags)):
  		if tags[i][1] in verb_tags:
  			mask_indexes.append(i)
  	for ind in mask_indexes:
  		tokenized_sent_2[ind] = '[MASK]'
  	tokenized_text = ["[CLS]"] + tokenized_sent + ["[SEP]"] + tokenized_sent_2 + ["[SEP]"]
  	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  	segments_ids = [0] + [0 for i in range(len(tokenized_sent))] + [1] + [1 for i in range(len(tokenized_sent))] + [1]
  	# Convert inputs to PyTorch tensors
  	tokens_tensor = torch.tensor([indexed_tokens])
  	segments_tensors = torch.tensor([segments_ids])
  	tokens_tensor = tokens_tensor.to('cuda')
  	segments_tensors = segments_tensors.to('cuda')
  	#Predict all tokens
  	with torch.no_grad():
  		outputs = model(tokens_tensor, token_type_ids=segments_tensors)
  		predictions = outputs[0]
  	needed_pred = 3
  	suggestions = dict()
  	# for ind in mask_indexes:
  	final_dict = {}
  	# print(mask_indexes)
  	for ind in range(len(tokenized_sent)):
  		if ind not in mask_indexes:
  			final_dict[ind] = [tokenized_sent[ind] , []]
  			continue
  		ind_n = len(tokenized_sent) + 2 + ind
  		mask_sorted = sorted(predictions[0, ind_n], reverse=True)
  		suggestions[tokenized_sent[ind]] = []
  		final_dict[ind] = [tokenized_sent[ind],[]]
  		# print(ind)
  		# print(final_dict)
  		for i in range(1,needed_pred+1):
  			pred_ind = (predictions[0, ind_n] == mask_sorted[i]).nonzero().item()
  			pred_token = tokenizer.convert_ids_to_tokens([pred_ind])[0]
  			suggestions[tokenized_sent[ind]].append(pred_token)
  			final_dict[ind][1].append(pred_token)
  	return final_dict

def get_prob(query ,dictionary, tokens, types):
  if query in dictionary:
    return float(dictionary[query]+1)/(tokens + types)
  else:
    return float(1)/(tokens + types)
def cal_mi(sub,verb,obj, score_obj):
  prob_svo = get_prob((sub,verb,obj), score_obj['svo'], score_obj['token_svo'],score_obj['type_svo'])
  prob_sub = get_prob(sub, score_obj['sub'], score_obj['token_sub'],score_obj['type_sub'])
  prob_obj = get_prob(obj, score_obj['obj'], score_obj['token_obj'],score_obj['type_obj'])
  prob_verb = get_prob(verb, score_obj['verb'], score_obj['token_verb'],score_obj['type_verb'])
  prob_sv = get_prob((sub,verb), score_obj['sv'], score_obj['token_sv'],score_obj['type_sv'])
  prob_vo = get_prob((verb,obj), score_obj['vo'], score_obj['token_vo'],score_obj['type_vo'])
    
  mi_svo = prob_svo/(prob_sub*prob_verb*prob_obj)
  mi_sv = prob_sv/(prob_sub*prob_verb)
  mi_vo = prob_vo/(prob_verb*prob_obj)
  
  return mi_svo/(1+mi_svo), mi_sv/(mi_sv+1), mi_vo/(1+mi_vo)
def give_verb_score(sub, verb, obj, verb_t, score_obj):
  if verb in score_obj['verb'] and verb_t in score_obj['verb']:
    
    score = 0
    s_svo, s_sv, s_vo = cal_mi(sub,verb,obj, score_obj)
    s_svto, s_svt, s_vto = cal_mi(sub,verb_t,obj, score_obj)
    
    score+=(s_svto-s_svo)/(s_svto+s_svo)
    if s_svto > s_svo+0.01*s_svo*(1-s_svo):
      return score
    score+=(s_svt-s_sv)/(s_vto+s_vo)
    score+=(s_svt-s_sv)/(s_vto+s_vo)
    
    return score/3
    
  else:
    return -1
def update_score_obj(score_obj, sub, verb, obj):
  lemmatizer = WordNetLemmatizer()
  sub_l = lemmatizer.lemmatize(sub)
  verb_l = lemmatizer.lemmatize(verb)
  obj_l = lemmatizer.lemmatize(obj)

  score_obj['token_svo']+=48
  score_obj['token_sv']+=48
  score_obj['token_vo']+=48
  score_obj['token_sub']+=48
  score_obj['token_verb']+=48
  score_obj['token_obj']+=48
  try:
    score_obj['svo'][(sub_l, verb_l, obj_l)]+=48
  except KeyError:
    score_obj['type_svo']+=1
    score_obj['svo'][(sub_l, verb_l, obj_l)]=48
  try:
    score_obj['sv'][(sub_l, verb_l)]+=48
  except KeyError:
    score_obj['type_sv']+=1
    score_obj['sv'][(sub_l, verb_l)]=48
  try:
    score_obj['vo'][(verb_l, obj_l)]+=48
  except KeyError:
    score_obj['type_vo']+=1
    score_obj['vo'][(verb_l, obj_l)]=48
  try:
    score_obj['sub'][sub_l]+=48
  except KeyError:
    score_obj['type_sub']+=1
    score_obj['sub'][sub_l]=48
  try:
    score_obj['verb'][verb_l]+=48
  except KeyError:
    score_obj['type_verb']+=1
    score_obj['verb'][verb_l]=48
  try:
    score_obj['obj'][obj_l]+=48
  except KeyError:
    score_obj['type_obj']+=1
    score_obj['obj'][obj_l]=48
  
  with open('score_obj_simple', 'wb') as f:
    pickle.dump(score_obj, f)
  print(score_obj['svo'][(sub_l, verb_l, obj_l)])
def get_verb_list(sub, verb, obj):
  lemmatizer = WordNetLemmatizer()
  with open('score_obj_simple', 'rb') as f:
    score_obj = pickle.load(f)
  synonyms = set()
  vtok = verb.split()
  vv = lemmatizer.lemmatize(vtok[-1].lower())
  for ss in wordnet.synsets(vv):
    for l in ss.lemmas():
      synonyms.add(l.name())

  score = []
  for vt in synonyms:
    vtok[-1] = vt
    verb_t = ' '.join(vtok)

    score.append(
    	give_verb_score(sub, verb, obj, verb_t, score_obj,)
    	)

  verb_list = [j[1] for j in sorted(list(zip(score, list(synonyms))), reverse=True)]

  return verb_list[:5]

def verb_replce_frontend(text, verb_replacement, svos, svos2):
  lst = text.split()
  original_verbs = []
  for i in range(len(svos)):
    original_verbs.append(svos2[i][1])
  verb_replacement_dict = {}
  for i in range(len(svos)):
    ind = lst.index(original_verbs[i])
    #verb_found[original_verbs[i]] = ind
    lst[ind] =  "MASK"
    verb_replacement_dict[ind] = verb_replacement[i]
  return verb_replacement_dict

def create_view_tensor(object):
    source = r"{% extends 'SST/tensorreplacementstemp.html' %}{% block sent %}"
    for i in object.values():
        if i[1]==[]:
            source+=i[0]+"&nbsp;"
            continue
        source+="""
        <div class="dropdown">
        <button class="dropbtn" ><span id="org_s"></span>{}</button>&nbsp;
        <div class="dropdown-content" id="dropdown_list">;
        """.format(i[0])
        for rep in i[1]:
            source+="""<div href=""><span>{}</span></div>""".format(rep)
        source+="</div></div>"
    source+=r"{% endblock %}"
    return source

def create_view_bert(object):
    source = r"{% extends 'SST/bertreplacementstemp.html' %}{% block sent %}"
    for i in object.values():
        if i[1]==[]:
            source+=i[0]+"&nbsp;"
            continue
        source+="""
        <div class="dropdown">
        <button class="dropbtn" ><span id="org_s"></span>{}</button>&nbsp;
        <div class="dropdown-content" id="dropdown_list">
        """.format(i[0])
        for rep in i[1]:
            source+="""<div href=""><span>{}</span></div>""".format(rep)
        source+="</div></div>"
    source+=r"{% endblock %}"
    return source

def create_view_goodenglish(object):
    source = r"{% extends 'SST/goodenglishreplacementstemp.html' %}{% block sent %}"
    for i in object.values():
        if i[1]==[]:
            source+=i[0]+"&nbsp;"
            continue
        source+="""
        <div class="dropdown">
        <button class="dropbtn" ><span id="org_s"></span>{}</button>&nbsp;
        <div class="dropdown-content" id="dropdown_list"> 
        """.format(i[0])
        for rep in i[1]:
            source+="""<div href=""><span>{}</span></div>""".format(rep)
        source+="</div></div>"
    source+=r"{% endblock %}"
    return source

def create_view_goodenglishup(object):
    source = r"{% extends 'SST/improvegoodenglish.html' %}{% block sent %}"
    for i in object.values():
        if i[1]==[]:
            source+=i[0]+"&nbsp;"
            continue
        source+="""
        <div class="dropdown">
        <button class="dropbtn" ><span id="org_s"></span>{}</button>&nbsp;
        <div class="dropdown-content" id="dropdown_list"> 
        """.format(i[0])
        for rep in i[1]:
            source+="""<div href=""><span>{}</span></div>""".format(rep)
        source+="</div></div>"
    source+=r"{% endblock %}"
    return source


def enter(request):
    return render(request, 'SST/enter.html')

@csrf_exempt
def verbreplacements_tensor(request):
    if request.method == 'POST':
        sentence = request.POST.get("sentence")
        try:
            text = nlp(sentence)
            svos = list(textacy.extract.subject_verb_object_triples(text))
            svos2 = list(textacy.extract.subject_verb_object_triples(text))
            for i in range(len(svos)):
                svos[i]=list(svos[i])
                for j in range(len(svos[i])):
                    svos[i][j] = str(svos[i][j])
                    svos[i][j] = lemmatizer.lemmatize(svos[i][j])
                svos[i] = tuple(svos[i])
            # print(svos)
            for i in range(len(svos2)):
                svos2[i]=list(svos2[i])
                for j in range(len(svos2[i])):
                    svos2[i][j] = str(svos2[i][j])
                svos2[i] = tuple(svos2[i])
            # print(svos2)
            replacements =  []
            for i in range(len(svos)):
                replacements.append(verb_replacements(svos[i][0],svos[i][1],svos[i][2]))
            # print(replacements)
            olddict = verb_replce_frontend(sentence, replacements, svos, svos2)
            final_dict= {}
            lst = sentence.split()
            # print(olddict)
            for i in range(len(lst)):
                if i in olddict.keys():
                    final_dict[i] = [lst[i], olddict[i]]
                else:
                    final_dict[i] = [lst[i], []]
            num = 0
            for key in final_dict.keys():
                num += len(final_dict[key][1])
            if (num==0):
                return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(\nIt requires some more data for training to be able to handle complex cases.')
            with open('SST/templates/SST/tensorreplacements.html', 'w') as f:
                f.write(create_view_tensor(final_dict))
            # source=create_view(final_dict)
            # outlist = []
            # for key in final_dict.keys():
                # outlist.append(final_dict[key])
            # context = {'outdict' : final_dict, 'sentence' : sentence}
            # print(final_dict)
            return render(request, 'SST/tensorreplacements.html')
        except:
            return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(\nIt requires some more data for training to be able to handle complex cases.')

    return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(\nIt requires some more data for training to be able to handle complex cases.')

@csrf_exempt
def verbreplacements_goodenglish(request):
    if request.method == 'POST':
        sentence = request.POST.get("sentence")
        nlp = spacy.load('en_core_web_sm')
        text = nlp(sentence)
        try:
            svos = list(textacy.extract.subject_verb_object_triples(text))
            svos2 = list(textacy.extract.subject_verb_object_triples(text))
            print('svos')
            print(svos)
            for i in range(len(svos)):
                svos[i]=list(svos[i])
                for j in range(len(svos[i])):
                    svos[i][j] = str(svos[i][j])
                    svos[i][j] = lemmatizer.lemmatize(svos[i][j])
                svos[i] = tuple(svos[i])
            print('changed svos')
            print(svos)
            for i in range(len(svos2)):
                svos2[i]=list(svos2[i])
                for j in range(len(svos2[i])):
                    svos2[i][j] = str(svos2[i][j])
                svos2[i] = tuple(svos2[i])
            # print(svos2)
            replacements =  []
            for i in range(len(svos)):
                replacements.append(get_verb_list(svos[i][0],svos[i][1],svos[i][2]))
            # print(replacements)
            olddict = verb_replce_frontend(sentence, replacements, svos, svos2)
            final_dict= {}
            lst = sentence.split()
            # print(olddict)
            for i in range(len(lst)):
                if i in olddict.keys():
                    final_dict[i] = [lst[i], olddict[i]]
                else:
                    final_dict[i] = [lst[i], []]
            num = 0
            for key in final_dict.keys():
                num += len(final_dict[key][1])
            if (num==0):
                return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(\nIt requires some more data for training to be able to handle complex cases.')
            with open('SST/templates/SST/goodenglishreplacements.html', 'w') as f:
                f.write(create_view_goodenglish(final_dict))
            with open('SST/templates/SST/improvegoodenglishr.html', 'w') as f:
                f.write(create_view_goodenglishup(final_dict))
            # source=create_view(final_dict)
            # outlist = []
            # for key in final_dict.keys():
                # outlist.append(final_dict[key])
            # context = {'outdict' : final_dict, 'sentence' : sentence}
            # print(final_dict)
            return render(request, 'SST/goodenglishreplacements.html')
        except:
            return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(\nIt requires some more data for training to be able to handle complex cases.')
    return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(\nIt requires some more data for training to be able to handle complex cases.')

@csrf_exempt
def verbreplacements_bert(request):
    if request.method == 'POST':
        sentence = request.POST.get("sentence")
        try:
            final_dict = pp_input(sentence)
            num = 0
            for key in final_dict.keys():
                num += len(final_dict[key][1])
            if num == 0:
                return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(')
            with open('SST/templates/SST/bertreplacements.html', 'w') as f:
                f.write(create_view_bert(final_dict))
            return render(request, 'SST/bertreplacements.html') 
        except:
            return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(')
    return HttpResponse('Sorry! This model could not find any replacements for your input sentence :(')

@csrf_exempt
def improvegoodenglish(request):
    return render(request, 'SST/improvegoodenglishr.html')

def onlinelearning(request):
    if request.method == 'POST':
        sentence = request.POST.get("sentence")
        print(sentence, "faltu")
        nlp = spacy.load('en_core_web_sm')
        text = nlp(sentence)
        svos = list(textacy.extract.subject_verb_object_triples(text))
        for i in range(len(svos)):
            svos[i]=list(svos[i])
            for j in range(len(svos[i])):
                svos[i][j] = str(svos[i][j])
                svos[i][j] = lemmatizer.lemmatize(svos[i][j])
            svos[i] = tuple(svos[i])
        with open('score_obj_simple', 'rb') as f:
            score_obj = pickle.load(f)
        print(svos, "faltu")
        for svo in svos:
            update_score_obj(score_obj, svo[0], svo[1], svo[2])

        return HttpResponse('Thanks! Good English Model Learning...')
    return HttpResponse('Something went wrong')