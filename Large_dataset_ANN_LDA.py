import numpy as np
from itertools import chain
import math
import pandas as pd 
import matplotlib.pyplot as plt
import time

st_time = time.time()
data = pd.read_csv('csv_result.csv')

mylist = list(data['class'].unique())

df_norm = data[['id1','id2','id3','id4','id5','id6','id7','id8','id9','id10','id11','id12','id13','id14','id15','id16','id17','id18','id19','id20','id21','id22','id23','id24','id25','id26','id27','id28','id29','id30','id31','id32','id33','id34','id35','id36','id37','id38','id39','id40','id41','id42','id43','id44','id45','id46','id47','id48','id49','id50','id51','id52','id53','id54','id55','id56','id57','id58','id59','id60','id61','id62','id63','id64','id65','id66','id67','id68','id69','id70','id71','id72','id73','id74','id75','id76','id77','id78','id79','id80','id81','id82','id83','id84','id85','id86','id87','id88','id89','id90','id91','id92','id93','id94','id95','id96','id97','id98','id99','id100','id101','id102','id103','id104','id105','id106','id107','id108','id109','id110','id111','id112','id113','id114','id115','id116','id117','id118','id119','id120','id121','id122','id123','id124','id125','id126','id127','id128','id129','id130','id131','id132','id133','id134','id135','id136','id137','id138','id139','id140','id141','id142','id143','id144','id145','id146','id147','id148','id149','id150','id151','id152','id153','id154','id155','id156','id157','id158','id159','id160','id161','id162','id163','id164','id165','id166','id167','id168','id169','id170','id171','id172','id173','id174','id175','id176','id177','id178','id179','id180','id181','id182','id183','id184','id185','id186','id187','id188','id189','id190','id191','id192','id193','id194','id195','id196','id197','id198','id199','id200','id201','id202','id203','id204','id205','id206','id207','id208','id209','id210','id211','id212','id213','id214','id215','id216','id217','id218','id219','id220','id221','id222','id223','id224','id225','id226','id227','id228','id229','id230','id231','id232','id233','id234','id235','id236','id237','id238','id239','id240','id241','id242','id243','id244','id245','id246','id247','id248','id249','id250','id251','id252','id253','id254','id255','id256','id257','id258','id259','id260','id261','id262','id263','id264','id265','id266','id267','id268','id269','id270','id271','id272','id273','id274','id275','id276','id277','id278','id279','id280','id281','id282','id283','id284','id285','id286','id287','id288','id289','id290','id291','id292','id293','id294','id295','id296','id297','id298','id299','id300','id301','id302','id303','id304','id305','id306','id307','id308','id309','id310','id311','id312','id313','id314','id315','id316','id317','id318','id319','id320','id321','id322','id323','id324','id325','id326','id327','id328','id329','id330','id331','id332','id333','id334','id335','id336','id337','id338','id339','id340','id341','id342','id343','id344','id345','id346','id347','id348','id349','id350','id351','id352','id353','id354','id355','id356','id357','id358','id359','id360','id361','id362','id363','id364','id365','id366','id367','id368','id369','id370','id371','id372','id373','id374','id375','id376','id377','id378','id379','id380','id381','id382','id383','id384','id385','id386','id387','id388','id389','id390','id391','id392','id393','id394','id395','id396','id397','id398','id399','id400','id401','id402','id403','id404','id405','id406','id407','id408','id409','id410','id411','id412','id413','id414','id415','id416','id417','id418','id419','id420','id421','id422','id423','id424','id425','id426','id427','id428','id429','id430','id431','id432','id433','id434','id435','id436','id437','id438','id439','id440','id441','id442','id443','id444','id445','id446','id447','id448','id449','id450','id451','id452','id453','id454','id455','id456','id457','id458','id459','id460','id461','id462','id463','id464','id465','id466','id467','id468','id469','id470','id471','id472','id473','id474','id475','id476','id477','id478','id479','id480','id481','id482','id483','id484','id485','id486','id487','id488','id489','id490','id491','id492','id493','id494','id495','id496','id497','id498','id499','id500','id501','id502','id503','id504','id505','id506','id507','id508','id509','id510','id511','id512','id513','id514','id515','id516','id517','id518','id519','id520','id521','id522','id523','id524','id525','id526','id527','id528','id529','id530','id531','id532','id533','id534','id535','id536','id537','id538','id539','id540','id541','id542','id543','id544','id545','id546','id547','id548','id549','id550','id551','id552','id553','id554','id555','id556','id557','id558','id559','id560','id561','id562','id563','id564','id565','id566','id567','id568','id569','id570','id571','id572','id573','id574','id575','id576','id577','id578','id579','id580','id581','id582','id583','id584','id585','id586','id587','id588','id589','id590','id591','id592','id593','id594','id595','id596','id597','id598','id599','id600','id601','id602','id603','id604','id605','id606','id607','id608','id609','id610','id611','id612','id613','id614','id615','id616','id617','id618','id619','id620','id621','id622','id623','id624','id625','id626','id627','id628','id629','id630','id631','id632','id633','id634','id635','id636','id637','id638','id639','id640','id641','id642','id643','id644','id645','id646','id647','id648','id649','id650','id651','id652','id653','id654','id655','id656','id657','id658','id659','id660','id661','id662','id663','id664','id665','id666','id667','id668','id669','id670','id671','id672','id673','id674','id675','id676','id677','id678','id679','id680','id681','id682','id683','id684','id685','id686','id687','id688','id689','id690','id691','id692','id693','id694','id695','id696','id697','id698','id699','id700','id701','id702','id703','id704','id705','id706','id707','id708','id709','id710','id711','id712','id713','id714','id715','id716','id717','id718','id719','id720','id721','id722','id723','id724','id725','id726','id727','id728','id729','id730','id731','id732','id733','id734','id735','id736','id737','id738','id739','id740','id741','id742','id743','id744','id745','id746','id747','id748','id749','id750','id751','id752','id753','id754','id755','id756','id757','id758','id759','id760','id761','id762','id763','id764','id765','id766','id767','id768','id769','id770','id771','id772','id773','id774','id775','id776','id777','id778','id779','id780','id781','id782','id783','id784','id785','id786','id787','id788','id789','id790','id791','id792','id793','id794','id795','id796','id797','id798','id799','id800','id801','id802','id803','id804','id805','id806','id807','id808','id809','id810','id811','id812','id813','id814','id815','id816','id817','id818','id819','id820','id821','id822','id823','id824','id825','id826','id827','id828','id829','id830','id831','id832','id833','id834','id835','id836','id837','id838','id839','id840','id841','id842','id843','id844','id845','id846','id847','id848','id849','id850','id851','id852','id853','id854','id855','id856','id857','id858','id859','id860','id861','id862','id863','id864','id865','id866','id867','id868','id869','id870','id871','id872','id873','id874','id875','id876','id877','id878','id879','id880','id881','id882','id883','id884','id885','id886','id887','id888','id889','id890','id891','id892','id893','id894','id895','id896','id897','id898','id899','id900','id901','id902','id903','id904','id905','id906','id907','id908','id909','id910','id911','id912','id913','id914','id915','id916','id917','id918','id919','id920','id921','id922','id923','id924','id925','id926','id927','id928','id929','id930','id931','id932','id933','id934','id935','id936','id937','id938','id939','id940','id941','id942','id943','id944','id945','id946','id947','id948','id949','id950','id951','id952','id953','id954','id955','id956','id957','id958','id959','id960','id961','id962','id963','id964','id965','id966','id967','id968','id969','id970','id971','id972','id973','id974','id975','id976','id977','id978','id979','id980','id981','id982','id983','id984','id985','id986','id987','id988','id989','id990','id991','id992','id993','id994','id995','id996','id997','id998','id999','id1000','id1001','id1002','id1003','id1004','id1005','id1006','id1007','id1008','id1009','id1010','id1011','id1012','id1013','id1014','id1015','id1016','id1017','id1018','id1019','id1020','id1021','id1022','id1023','id1024','id1025','id1026','id1027','id1028','id1029','id1030','id1031','id1032','id1033','id1034','id1035','id1036','id1037','id1038','id1039','id1040','id1041','id1042','id1043','id1044','id1045','id1046','id1047','id1048','id1049','id1050','id1051','id1052','id1053','id1054','id1055','id1056','id1057','id1058','id1059','id1060','id1061','id1062','id1063','id1064','id1065','id1066','id1067','id1068','id1069','id1070','id1071','id1072','id1073','id1074','id1075','id1076','id1077','id1078','id1079','id1080','id1081','id1082','id1083','id1084','id1085','id1086','id1087','id1088','id1089','id1090','id1091','id1092','id1093','id1094','id1095','id1096','id1097','id1098','id1099','id1100','id1101','id1102','id1103','id1104','id1105','id1106','id1107','id1108','id1109','id1110','id1111','id1112','id1113','id1114','id1115','id1116','id1117','id1118','id1119','id1120','id1121','id1122','id1123','id1124','id1125','id1126','id1127','id1128','id1129','id1130','id1131','id1132','id1133','id1134','id1135','id1136','id1137','id1138','id1139','id1140','id1141','id1142','id1143','id1144','id1145','id1146','id1147','id1148','id1149','id1150','id1151','id1152','id1153','id1154','id1155','id1156','id1157','id1158','id1159','id1160','id1161','id1162','id1163','id1164','id1165','id1166','id1167','id1168','id1169','id1170','id1171','id1172','id1173','id1174','id1175','id1176','id1177','id1178','id1179','id1180','id1181','id1182','id1183','id1184','id1185','id1186','id1187','id1188','id1189','id1190','id1191','id1192','id1193','id1194','id1195','id1196','id1197','id1198','id1199','id1200','id1201','id1202','id1203','id1204','id1205','id1206','id1207','id1208','id1209','id1210','id1211','id1212','id1213','id1214','id1215','id1216','id1217','id1218','id1219','id1220','id1221','id1222','id1223','id1224','id1225','id1226','id1227','id1228','id1229','id1230','id1231','id1232','id1233','id1234','id1235','id1236','id1237','id1238','id1239','id1240','id1241','id1242','id1243','id1244','id1245','id1246','id1247','id1248','id1249','id1250','id1251','id1252','id1253','id1254','id1255','id1256','id1257','id1258','id1259','id1260','id1261','id1262','id1263','id1264','id1265','id1266','id1267','id1268','id1269','id1270','id1271','id1272','id1273','id1274','id1275','id1276','id1277','id1278','id1279','id1280','id1281','id1282','id1283','id1284','id1285','id1286','id1287','id1288','id1289','id1290','id1291','id1292','id1293','id1294','id1295','id1296','id1297','id1298','id1299','id1300','id1301','id1302','id1303','id1304','id1305','id1306','id1307','id1308','id1309','id1310','id1311','id1312','id1313','id1314','id1315','id1316','id1317','id1318','id1319','id1320','id1321','id1322','id1323','id1324','id1325','id1326','id1327','id1328','id1329','id1330','id1331','id1332','id1333','id1334','id1335','id1336','id1337','id1338','id1339','id1340','id1341','id1342','id1343','id1344','id1345','id1346','id1347','id1348','id1349','id1350','id1351','id1352','id1353','id1354','id1355','id1356','id1357','id1358','id1359','id1360','id1361','id1362','id1363','id1364','id1365','id1366','id1367','id1368','id1369','id1370','id1371','id1372','id1373','id1374','id1375','id1376','id1377','id1378','id1379','id1380','id1381','id1382','id1383','id1384','id1385','id1386','id1387','id1388','id1389','id1390','id1391','id1392','id1393','id1394','id1395','id1396','id1397','id1398','id1399','id1400','id1401','id1402','id1403','id1404','id1405','id1406','id1407','id1408','id1409','id1410','id1411','id1412','id1413','id1414','id1415','id1416','id1417','id1418','id1419','id1420','id1421','id1422','id1423','id1424','id1425','id1426','id1427','id1428','id1429','id1430','id1431','id1432','id1433','id1434','id1435','id1436','id1437','id1438','id1439','id1440','id1441','id1442','id1443','id1444','id1445','id1446','id1447','id1448','id1449','id1450','id1451','id1452','id1453','id1454','id1455','id1456','id1457','id1458','id1459','id1460','id1461','id1462','id1463','id1464','id1465','id1466','id1467','id1468','id1469','id1470','id1471','id1472','id1473','id1474','id1475','id1476','id1477','id1478','id1479','id1480','id1481','id1482','id1483','id1484','id1485','id1486','id1487','id1488','id1489','id1490','id1491','id1492','id1493','id1494','id1495','id1496','id1497','id1498','id1499','id1500','id1501','id1502','id1503','id1504','id1505','id1506','id1507','id1508','id1509','id1510','id1511','id1512','id1513','id1514','id1515','id1516','id1517','id1518','id1519','id1520','id1521','id1522','id1523','id1524','id1525','id1526','id1527','id1528','id1529','id1530','id1531','id1532','id1533','id1534','id1535','id1536','id1537','id1538','id1539','id1540','id1541','id1542','id1543','id1544','id1545','id1546','id1547','id1548','id1549','id1550','id1551','id1552','id1553','id1554','id1555','id1556','id1557','id1558','id1559','id1560','id1561','id1562','id1563','id1564','id1565','id1566','id1567','id1568','id1569','id1570','id1571','id1572','id1573','id1574','id1575','id1576','id1577','id1578','id1579','id1580','id1581','id1582','id1583','id1584','id1585','id1586','id1587','id1588','id1589','id1590','id1591','id1592','id1593','id1594','id1595','id1596','id1597','id1598','id1599','id1600','id1601','id1602','id1603','id1604','id1605','id1606','id1607','id1608','id1609','id1610','id1611','id1612','id1613','id1614','id1615','id1616','id1617','id1618','id1619','id1620','id1621','id1622','id1623','id1624','id1625','id1626','id1627','id1628','id1629','id1630','id1631','id1632','id1633','id1634','id1635','id1636','id1637','id1638','id1639','id1640','id1641','id1642','id1643','id1644','id1645','id1646','id1647','id1648','id1649','id1650','id1651','id1652','id1653','id1654','id1655','id1656','id1657','id1658','id1659','id1660','id1661','id1662','id1663','id1664','id1665','id1666','id1667','id1668','id1669','id1670','id1671','id1672','id1673','id1674','id1675','id1676','id1677','id1678','id1679','id1680','id1681','id1682','id1683','id1684','id1685','id1686','id1687','id1688','id1689','id1690','id1691','id1692','id1693','id1694','id1695','id1696','id1697','id1698','id1699','id1700','id1701','id1702','id1703','id1704','id1705','id1706','id1707','id1708','id1709','id1710','id1711','id1712','id1713','id1714','id1715','id1716','id1717','id1718','id1719','id1720','id1721','id1722','id1723','id1724','id1725','id1726','id1727','id1728','id1729','id1730','id1731','id1732','id1733','id1734','id1735','id1736','id1737','id1738','id1739','id1740','id1741','id1742','id1743','id1744','id1745','id1746','id1747','id1748','id1749','id1750','id1751','id1752','id1753','id1754','id1755','id1756','id1757','id1758','id1759','id1760','id1761','id1762','id1763','id1764','id1765','id1766','id1767','id1768','id1769','id1770','id1771','id1772','id1773','id1774','id1775','id1776','id1777','id1778','id1779','id1780','id1781','id1782','id1783','id1784','id1785','id1786','id1787','id1788','id1789','id1790','id1791','id1792','id1793','id1794','id1795','id1796','id1797','id1798','id1799','id1800','id1801','id1802','id1803','id1804','id1805','id1806','id1807','id1808','id1809','id1810','id1811','id1812','id1813','id1814','id1815','id1816','id1817','id1818','id1819','id1820','id1821','id1822','id1823','id1824','id1825','id1826','id1827','id1828','id1829','id1830','id1831','id1832','id1833','id1834','id1835','id1836','id1837','id1838','id1839','id1840','id1841','id1842','id1843','id1844','id1845','id1846','id1847','id1848','id1849','id1850','id1851','id1852','id1853','id1854','id1855','id1856','id1857','id1858','id1859','id1860','id1861','id1862','id1863','id1864','id1865','id1866','id1867','id1868','id1869','id1870','id1871','id1872','id1873','id1874','id1875','id1876','id1877','id1878','id1879','id1880','id1881','id1882','id1883','id1884','id1885','id1886','id1887','id1888','id1889','id1890','id1891','id1892','id1893','id1894','id1895','id1896','id1897','id1898','id1899','id1900','id1901','id1902','id1903','id1904','id1905','id1906','id1907','id1908','id1909','id1910','id1911','id1912','id1913','id1914','id1915','id1916','id1917','id1918','id1919','id1920','id1921','id1922','id1923','id1924','id1925','id1926','id1927','id1928','id1929','id1930','id1931','id1932','id1933','id1934','id1935','id1936','id1937','id1938','id1939','id1940','id1941','id1942','id1943','id1944','id1945','id1946','id1947','id1948','id1949','id1950','id1951','id1952','id1953','id1954','id1955','id1956','id1957','id1958','id1959','id1960','id1961','id1962','id1963','id1964','id1965','id1966','id1967','id1968','id1969','id1970','id1971','id1972','id1973','id1974','id1975','id1976','id1977','id1978','id1979','id1980','id1981','id1982','id1983','id1984','id1985','id1986','id1987','id1988','id1989','id1990','id1991','id1992','id1993','id1994','id1995','id1996','id1997','id1998','id1999','id2000']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
target = data[['class']].replace(['Normal','Tumor'],[0,1])

df = pd.concat([df_norm, target], axis=1)

train_test_per = 80/100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train1 = df[df.train == 1]
train = train1.drop('train', axis=1).sample(frac=1)

X = train.values[:,:2000]
y = train.values[:,2000:2001]

test = df[df.train == 0]
test = test.drop('train', axis=1).sample(frac=1)

X1 = test.values[:,:2000]
y1 = test.values[:,2000:2001]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 10)
In_train = lda.fit_transform(X,y)
In_test = lda.transform(X1)
explained_variance = lda.explained_variance_ratio_


num_inputs = len(In_train[0])
hidden_layer_neurons = 5
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
#np.random.seed(5)
#learning_rate = 0.01
learning_rate = np.random.uniform(0,0.03)
print(learning_rate)

er = [0]*1000
for epoch in range(1000):
    l1 = 1/(1 + np.exp(-(np.dot(In_train, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    x = list(chain(*(abs(y - l2))))
    er[epoch] = pow((np.mean([math.sqrt(i) for i in x])),2)
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += In_train.T.dot(l1_delta) * learning_rate

epoch1 = list(range(epoch+1))
ax = plt.subplot()
p1, =plt.plot(epoch1,er, '-.', color='black', label= 'ANN with ICA', lw=2)
plt.xlabel("Number of Epochs") 
plt.ylabel("Error")
ax.legend()
plt.show()     

y1 = (y1==1)

l1 = 1/(1 + np.exp(-(np.dot(In_test, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

yp = (l2 >= 0.5) # prediction
res = yp == y1
correct = np.sum(res)/len(res)

y1 = list(chain(*(y1)))
yp = list(chain(*(yp)))

y_actu = pd.Series(y1, name='Actual')
y_pred = pd.Series(yp, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

prec = (df_confusion[True][True]/(df_confusion[True][True]+df_confusion[True][False]))
recall = (df_confusion[True][True]/(df_confusion[True][True]+df_confusion[False][False]))
f_measure = (2*prec*recall)/(prec+recall)

testres = test[['class']].replace([0,1], ['Normal','Tumor'])
testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1],['Normal','Tumor'])

print(testres)
print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')
print('Error:',np.mean(er))
print('Precision:',prec)
print('Recall:',recall)
print('F-Measure:',f_measure)
print("Time: %s  seconds"%(time.time() - st_time))