import json
import base64
import mysql.connector
import sys
import math
import time
from random import randint
import numpy as np
import scipy as sp
from datasketch import MinHashLSHForest, MinHash
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from collections import Counter
import itertools

np.set_printoptions(suppress=True)

ratingMap = {5: 'likes', 4: 'interested', 3: 'neutral', 2: 'dislikes', 1: 'not-interested'}


def convertRating(ratingTuple):
    mangaId = ratingTuple[2]
    status = ratingTuple[3]
    originalRating = ratingTuple[4]

    if originalRating is not None:
        if originalRating >= 7:
            return mangaId, 5  # 5
        if originalRating >= 5:
            return mangaId, 3  # 3
        else:  # if rating < 5
            return mangaId, 2  # 2
    if status == 'reading' or status == 'completed':
        return mangaId, 4  # 4
    if status == 'plan_to_read':
        return mangaId, 4  # 4
    if status == 'on_hold':
        return mangaId, 3  # 3
    if status == 'dropped':
        return mangaId, 2  # 2
    if status == 'not interested':
        return mangaId, 1
    return mangaId, 4  # 4  # happens if status and rating are none.
    # list of assumptions: if you finished it you liked it
    # if you are reading it or plan to read it you are interested in it
    # if you rated it we map that into liked, disliked, neutral
    # on hold maps to neutral (we cannot make any assumptions with on hold imo so neutral is best)
    # dropped maps to disliked since we assume there was something you disliked about it that caused you to drop it


def convertRating_forKnn(ratingTuple):
    convertedRating = convertRating(ratingTuple)
    return ratingTuple[1], convertedRating[0], convertedRating[1]  # userId, mangaId, convertedRatingValue


def satisfiesFilters(manga, filters):
    # if I want include check for include exclude overlap
    if not (filters[0][0] <= manga[1] <= filters[0][1]):
        return False  # popularity
    if manga[2] is not None:
        if not (filters[1][0] <= manga[2] <= filters[1][1]):
            return False  # releaseDate
    if manga[3] is not None:
        if not (filters[2][0] <= manga[3] <= filters[2][1]):
            return False  # chapterCount
    for i in range(len(filters[3])):
        if filters[3][i] is True and manga[4 + i] == 1:
            return False  # exclude status
    for i in range(len(filters[4])):
        if filters[4][i] is True and manga[8 + i] == 0:
            return False  # include genre
    for i in range(len(filters[5])):
        if filters[5][i] is True and manga[26 + i] == 0:
            return False  # include theme
    for i in range(len(filters[6])):
        if filters[6][i] is True and manga[77 + i] == 0:
            return False  # include demographic
    for i in range(len(filters[7])):
        if filters[7][i] is True and manga[8 + i] == 1:
            return False  # exclude genre
    for i in range(len(filters[8])):
        if filters[8][i] is True and manga[26 + i] == 1:
            return False  # exclude theme
    for i in range(len(filters[9])):
        if filters[9][i] is True and manga[77 + i] == 1:
            return False  # exclude demographic
    return True


def getStatusSet():
    return ['On Hiatus', 'Finished', 'Publishing', 'Discontinued']


def getGenreSet():
    return ['Adventure', 'Comedy', 'Slice of Life', 'Boys Love', 'Sci-Fi', 'Action', 'Horror', 'Suspense', 'Girls Love',
            'Gourmet', 'Sports', 'Avant Garde', 'Supernatural', 'Fantasy', 'Romance', 'Ecchi', 'Drama', 'Mystery']


def getThemeSet():
    return ['Historical', 'Time Travel', 'Visual Arts', 'Military', 'Love Polygon', 'Mecha', 'Martial Arts', 'Racing',
            'Samurai', 'Strategy Game', 'CGDCT', 'Mythology', 'High Stakes Game', 'Idols (Male)', 'Reincarnation',
            'Pets', 'Team Sports', 'Workplace', 'Isekai', 'Gag Humor', 'Memoir', 'Harem', 'Villainess', 'Detective',
            'Performing Arts', 'Reverse Harem', 'Childcare', 'Otaku Culture', 'Mahou Shoujo', 'Anthropomorphic',
            'Survival', 'Magical Sex Shift', 'Music', 'Delinquents', 'Organized Crime', 'Adult Cast', 'Medical',
            'Showbiz', 'Crossdressing', 'Gore', 'Psychological', 'School', 'Combat Sports', 'Parody',
            'Romantic Subtext', 'Space', 'Iyashikei', 'Video Game', 'Educational', 'Vampire', 'Super Power']


def getDemographicSet():
    return ['Kids', 'Seinen', 'Shoujo', 'Josei', 'Shounen']


def encodeManga(manga, filters):
    mangaEncoded = []
    excludedIds = []
    statusSet = getStatusSet()
    genreSet = getGenreSet()
    themeSet = getThemeSet()
    demographicSet = getDemographicSet()
    for mangaInstance in manga:
        # make fill release date for None type,
        # uses 2008 as fill for when release date is none. 2008 is average release data and there are only 100 manga...
        # with no release date
        # rn chapterCount is set to 0 if none, I dont like this but idk what else to do (avg is 40)
        instanceData = [mangaInstance[0], mangaInstance[1], mangaInstance[4] if mangaInstance[4] is not None else 2008,
                        mangaInstance[5] if mangaInstance[5] is not None else 40]

        mangaInstanceStatusSet = mangaInstance[6].replace("\"", "")
        for i in range(len(statusSet)):
            if statusSet[i] == mangaInstanceStatusSet:
                instanceData.append(1)
            else:
                instanceData.append(0)
        mangaInstanceGenreSet = mangaInstance[7].replace("\"", "").split('|') if mangaInstance[7] is not None else []
        for i in range(len(genreSet)):
            if genreSet[i] in mangaInstanceGenreSet:
                instanceData.append(1)
            else:
                instanceData.append(0)
        mangaInstanceThemeSet = mangaInstance[8].replace("\"", "").split('|') if mangaInstance[8] is not None else []
        for i in range(len(themeSet)):
            if themeSet[i] in mangaInstanceThemeSet:
                instanceData.append(1)
            else:
                instanceData.append(0)
        mangaInstanceDemographicSet = mangaInstance[9].replace("\"", "").split('|') if mangaInstance[
                                                                                           9] is not None else []
        for i in range(len(demographicSet)):
            if demographicSet[i] in mangaInstanceDemographicSet:
                instanceData.append(1)
            else:
                instanceData.append(0)

        if satisfiesFilters(instanceData, filters):
            mangaEncoded.append(instanceData)
        else:
            excludedIds.append(mangaInstance[0])
    return mangaEncoded, excludedIds


def cosineSimilarity(vector1, vector2):
    sumXX, sumXY, sumYY = 0, 0, 0
    for i in range(len(vector1)):
        X = vector1[i]
        Y = vector2[i]
        sumXX += X * X
        sumYY += Y * Y
        sumXY += X * Y
    return sumXY / math.sqrt(sumXX * sumYY) if sumXX * sumYY > 0 else 0

def jaccardSimilarity(set1, set2):
    return float(len(set1.intersection(set2))) / float(len(set1.union(set2)))

def knn(myCursor, mangaIds, userId):
    k = 65  # i think 20 would be good
    # 130 is sqrt(num users) 65 is sqrt(num users)/2  <- both common choice of k

    rating_time = time.time()
    myCursor.execute("select * from ratings;")
    ratings = myCursor.fetchall()  # [x for x in myCursor]
    convertedRatings = [convertRating_forKnn(x) for x in ratings]
    # print("get ratings time is", time.time() - rating_time)
    # print(convertedRatings[0:20])
    myCursor.execute("select distinct userId from users;")
    userIds = myCursor.fetchall()
    userIds = [x[0] for x in userIds]
    knn_time = time.time()
    # user and manga dict are used to map from the user/manga ids to there index in the user_matrix
    mangaDict = dict()
    for index, value in enumerate(mangaIds):
        mangaDict[value] = index
    userDict = dict()
    for index, value in enumerate(userIds):
        userDict[value] = index
    userIndex = userDict[userId]

    # start of section unique to knn
    matrix_time = time.time()
    user_matrix = np.zeros((len(userIds), len(mangaIds)), dtype=np.int8)
    for i in range(len(convertedRatings)):
        user_matrix[userDict[convertedRatings[i][0]], mangaDict[convertedRatings[i][1]]] = convertedRatings[i][2]
    #print("generating user matrix took:", time.time() - matrix_time)
    # print(user_matrix)

    # slow 100s distance calculation without using sparse matrix
    # distance_time = time.time()
    # distances = np.zeros(shape=(len(user_matrix), 2))
    # for i in range(len(user_matrix)):
    #     distances[i][0] = userIds[i]  # or i or userDict[i]
    #     if i == userDict[userId]:
    #         continue  # the user will have a 0 sim to its self
    #     distances[i][1] = cosineSimilarity(user_matrix[userIndex], user_matrix[i])
    # print("distance time is:", time.time() - distance_time)
    # distances = distances[distances[:, 1].argsort()[::-1]]  # sort distances
    # print(distances[0:10])

    # I tried csr and csc.  csr: 18s  csc: 15.16s  coo: 14.6s
    convert_time = time.time()
    coo_user_matrix = sparse.coo_matrix(user_matrix)
    # csr_user_matrix = coo_user_matrix.tocsr()  # not noticeably faster
    #print("converting to sparse matrix took:", time.time() - convert_time)
    sparse_coo_time = time.time()
    similarities = cosine_similarity(coo_user_matrix)
    similarities_toUser = np.zeros(shape=(len(user_matrix), 2))
    for i in range(len(similarities[userDict[userId]])):
        similarities_toUser[i][0] = userIds[i]  # or i or userDict[i]
        if i == userDict[userId]:
            continue  # the user will have a 0 sim to its self
        similarities_toUser[i][1] = similarities[userIndex][i]
    #print("sparse build time is:", time.time() - sparse_coo_time)
    # print(similarities)
    # print(similarities_toUser[0:10])
    sort_time = time.time()
    # print(distances[0:10])
    similarities_toUser = similarities_toUser[similarities_toUser[:, 1].argsort()[::-1]]
    # print(similarities_toUser[0:10])
    #print("sort time is:", time.time() - sort_time)

    kNeighbors = []
    neighborsManga = []
    for x in range(k):
        kNeighbors.append(int(similarities_toUser[x][0]))
        neighborsManga.append(user_matrix[userDict[kNeighbors[x]]])
    #print("knn took:", time.time() - knn_time)

    LSH_time = time.time()
    num_permutations = 16  # ********** increase to improve results at cost of speed **********
    # 128 common default
    similarUsers = LSH(userId, userIds, userDict, convertedRatings, num_permutations, k)
    #print("LSH took:", time.time() - LSH_time)

    #print("knn:", kNeighbors)
    #print("LSH:", similarUsers)
    overlap = 0
    for i in range(len(kNeighbors)):
        if kNeighbors[i] in similarUsers:
            overlap += 1
    #print("knn and LSH had %s users overlap, out of %s aka %s percent" % (overlap, k, (overlap*100/k)))

    useLSH = True
    if useLSH:
        kNeighbors = similarUsers
        neighborsManga = []
        for i in range(len(similarUsers)):
            neighborsManga.append(user_matrix[userDict[similarUsers[i]]])

    return kNeighbors, neighborsManga


def LSH(userId, userIds, userDict, ratings, num_permutations, k):
    #print("starting LSH")
    # k is number of similar users to return
    # num_permutations is number of hash functions
    run_package_code = True

    set_time = time.time()
    user_sets = [set() for i in range(len(userDict))]  # create empty set for each user
    # print(user_sets)
    for rating in ratings:
        if rating[2] >= 3:  # only include if the rating is 5: 'likes', 4: 'interested', 3: 'neutral'
            user_sets[userDict[rating[0]]].add(rating[1])
    #print("creating sets took:", time.time() - set_time)

    if run_package_code:
        # ************** min hash generation using package code **************
        min_hash_gen_time = time.time()
        signature_matrix_package = []
        for user_set in user_sets:
            min_hash = MinHash(num_perm=num_permutations)
            for mangaId in user_set:
                min_hash.update(str(mangaId).encode('utf8'))
            signature_matrix_package.append(min_hash)
        #print("min hash gen time:", time.time() - min_hash_gen_time)
        # ************** min hash LSH forest generation using package code **************
        forest_gen_time = time.time()
        user_min_hash = None
        LSH_forest = MinHashLSHForest(num_perm=num_permutations)
        for i in range(len(signature_matrix_package)):  # should be equal to number of users
            if userIds[i] == userId:
                user_min_hash = signature_matrix_package[i]
                continue  # dont include the user in the forest
            LSH_forest.add(i, signature_matrix_package[i])  # stores user index not userId
        LSH_forest.index()
        #print("forest gen time:", time.time() - forest_gen_time)
        # for i in range(len(userDict)):
        # ************** get k neighbors using package code **************
        query_time = time.time()
        # user_min_hash = MinHash(num_perm=num_permutations)
        # for mangaId in user_sets[userDict[userId]]:
        #     user_min_hash.update(str(mangaId).encode('utf8'))
        similarIndices = np.array(LSH_forest.query(user_min_hash, k + int(k/10)))  # include extra results
        similarUsers = [[userIds[x], user_sets[x]] for x in similarIndices]
    else:
        # ************** min hash generation using my code **************
        min_hash_gen_time = time.time()
        signature_matrix = []
        max_val = (2 ** 32) - 1
        perms = [(randint(0, max_val), randint(0, max_val)) for i in range(num_permutations)]
        for user_set in user_sets:
            min_hash = minhash(user_set, num_permutations, perms)
            signature_matrix.append(min_hash)
        #print("min hash gen time:", time.time() - min_hash_gen_time)
        #print(signature_matrix)
        #print(signature_matrix[userDict[userId]])
        # ************** min hash LSH generation using my code **************
        forest_gen_time = time.time()
        user_min_hash = None
        lsh = {}
        bandSize = 2  # larger number will reduce collision. worked with 2 ######################################
        for i in range(len(signature_matrix)):  # should be equal to number of users
            if userIds[i] == userId:
                user_min_hash = signature_matrix[i]  # save user min hash
                continue  # dont include the user in the lsh
            # stores userId not user index
            for x in range(len(signature_matrix[i]) - bandSize):
                hashCode = ""
                for y in range(bandSize):
                    hashCode += str(signature_matrix[i][x+y])
                if hashCode in lsh:
                    lsh[hashCode].append(userIds[i])
                else:
                    lsh[hashCode] = [userIds[i]]
        #print(user_min_hash)
        #print("forest gen time:", time.time() - forest_gen_time)
        # ************** get k neighbors using my code **************
        query_time = time.time()
        # user_min_hash = MinHash(num_perm=num_permutations)
        # for mangaId in user_sets[userDict[userId]]:
        #     user_min_hash.update(str(mangaId).encode('utf8'))
        similarUserIdCounts = Counter()
        for x in range(len(user_min_hash) - bandSize):
            hashCode = ""
            for y in range(bandSize):
                hashCode += str(user_min_hash[x + y])
            if hashCode in lsh:
                for similarUserId in lsh[hashCode]:
                    similarUserIdCounts[similarUserId] += 1
        similarUserIdCounts = dict(sorted(similarUserIdCounts.items(), key=lambda item: item[1], reverse=True))
        # ^ sorts similar users by frequency they appeared in same lsh buckets
        #print(similarUserIdCounts)
        #print(list(similarUserIdCounts)[0])
        #print(len(similarUserIdCounts))

        similarUserIds = list(similarUserIdCounts)
        similarUsers = [[x, user_sets[userDict[x]]] for x in similarUserIds]
        #print("similarUsers", similarUsers)

    jaccardSims = np.array([[x[0], jaccardSimilarity(x[1], user_sets[userDict[userId]])] for x in similarUsers])
    #print(jaccardSims)
    jaccardSims = jaccardSims[jaccardSims[:, 1].argsort()[::-1]]
    #print(jaccardSims)
    #print(len(jaccardSims))
    #print("query time:", time.time() - query_time)

    return [int(jaccardSims[x][0]) for x in range(k)]

def minhash(user_set, num_permutations, perm_functions, prime=4294967311):  # no clue what to make prime 4294967311 429497
    # initialize a minhash vector of length num_permutations with positive infinity values
    vector = [float('inf') for i in range(num_permutations)]

    for val in user_set:

        # loop over each "permutation function"
        for perm_idx, perm_vals in enumerate(perm_functions):
            a, b = perm_vals

            # pass `val` through the `ith` permutation function
            output = (a * val + b) % prime

            # conditionally update the `ith` value of the vector
            if vector[perm_idx] > output:
                vector[perm_idx] = output

    # the returned vector represents the minimum hash of the user_set
    return vector


def recommend(userId: int, filters):
    dataBase = mysql.connector.connect(
        host="washington.uww.edu",
        user="stremmeltr18",
        passwd=base64.b64decode(b'dHM1NjEy').decode("utf-8"),
        database="manga_rec"
    )
    myCursor = dataBase.cursor()

    # create one hot encoded (and other data alterations) matrix of movie table
    # possible include no_genres/no_themes column (i dont think it would be good but idk)
    myCursor.execute("select * from manga;")
    manga = [x for x in myCursor]
    # 0:id, 1:popularity, 2: releaseDate, 3:chapterCount, 4-7:status, 8-25:genre, 26-76:theme, 77-81:demographic
    mapping = ['id', 'popularity', 'releaseDate',
               'chapterCount'] + getStatusSet() + getGenreSet() + getThemeSet() + getDemographicSet()
    mangaEncoded, excludedIds = encodeManga(manga, filters)
    for i in range(len(manga)):
        manga[i] = list(manga[i])
        del manga[i][3]  # drop description for ease of viewing output

    # get user's manga ratings
    myCursor.execute("select * from ratings where userId = %s;", [userId])
    ratings = [x for x in myCursor]
    convertedRatings = [convertRating(x) for x in ratings]
    ##print(convertedRatings)
    # #print('\n'.join([str(x) for x in convertedRatings]))

    # create table of encoded manga the user has rated
    userTable = []
    filteredRatings = []
    for i in range(len(convertedRatings)):
        for j in range(len(mangaEncoded)):
            if convertedRatings[i][0] == mangaEncoded[j][0]:  # only include manga that have not been filtered out
                userTable.append(mangaEncoded[j])  # [4:]only uses the one hot values for now
                filteredRatings.append(list(convertedRatings[i]))
                break
    ratedMangaIds = [i[0] for i in filteredRatings]
    # print('\n'.join([str(x) for x in userTable]))
    # print('\n'.join([str(filteredRatings[x]) + str(manga[[i[0] for i in manga].index(filteredRatings[x][0])]) for x in range(len(filteredRatings))])) # HELPFUL
    # print('\n'.join([str(mangaEncoded[[i[0] for i in mangaEncoded].index(filteredRatings[x][0])]) for x in range(len(filteredRatings))])) # HELPFUL

    # create a user preference vector (average or dot product of all user ratings)
    weightedTotal = [0] * len(userTable[0])  # aka number of features
    featureCounts = [0] * len(userTable[0])
    for i in range(len(userTable)):
        for j in range(len(userTable[i])):
            if userTable[i][j] is not None:
                weightedTotal[j] += userTable[i][j] * (filteredRatings[i][1] - 3)
                featureCounts[j] += (filteredRatings[i][1] - 3)
    userProfile = [0] * len(userTable[0])
    for i in range(len(weightedTotal)):
        userProfile[i] = (weightedTotal[i] / featureCounts[i]) if featureCounts != 0 else 0

    #print(dict(zip(mapping, userProfile)))  # HELPFUL

    # create recommendations
    similarityMeasures = np.zeros(shape=(len(mangaEncoded), 2))  # I think this is the right size
    # next 2 lines remove id for similarity calculation (it is not a feature) I will now try to remove them when sending the vectors to the cosign function
    # ********** testing ********** also removes popularity, release date, and chapter count
    mangaVectors = [mangaEncoded[i][4:] for i in range(len(mangaEncoded))]  # change 1 to 2 to remove popularity feature
    userVector = userProfile[4:]  # change 1 to 2 to remove popularity feature
    for i in range(len(similarityMeasures)):
        similarityMeasures[i][0] = mangaEncoded[i][0]
        similarityMeasures[i][1] = cosineSimilarity(userVector, mangaVectors[i])
    # print(similarityMeasures)
    similarityMeasures = similarityMeasures[similarityMeasures[:, 1].argsort()[::-1]]  # sort similarityMeasures
    # print(similarityMeasures)  # HELPFUL

    # get the manga rows that were recommended (and exclude those the user has already rated)
    similarManga = []
    for i in range(len(similarityMeasures)):
        if len(similarManga) >= 25:  # *********** may need to change to change size of result set ***********
            break
        for j in manga:
            if similarityMeasures[i][0] == j[0] and j[0] not in ratedMangaIds:
                similarManga.append(j)
    # print('\n'.join([str(x) for x in recommendedManga]))  # HELPFUL
    # print('\n'.join([str(similarityMeasures[[i[0] for i in similarityMeasures].index(x[0])][1]) + "\t" + str(x) for x in
    #                  similarManga]))  # HELPFUL

    #print("starting knn")
    similarUsers, similarUsers_matrixRow = knn(myCursor, [i[0] for i in manga], userId)
    myCursor.close()
    similarMangaIdCounts = Counter()
    for i in range(len(similarUsers_matrixRow)):
        for j in range(len(similarUsers_matrixRow[i])):
            if similarUsers_matrixRow[i][j] == 4 or similarUsers_matrixRow == 5:
                # if they liked or are interested in the manga at index j
                similarMangaIdCounts[manga[j][0]] += 1  # add the mangaId for that index to list

    # next line sorts by frequency then by id (which is basically ranking) (aka ties broken by cite wide rating)
    # similarMangaIdCounts = {val[0]: val[1] for val in sorted(similarMangaIdCounts.items(), key=lambda x: (-x[1], x[0]))}
    # next line sorts by frequency but keeps insertion order (aka ties broken by the order of similarity of the user the manga comes from)
    similarMangaIdCounts = dict(sorted(similarMangaIdCounts.items(), key=lambda item: item[1], reverse=True))
    # the idea behind the above line is to sightly reduce the bias for popular manga compared to the line three above this
    # print(similarMangaIdCounts)
    # filter out manga that dont satisfy filters
    for i in range(len(excludedIds)):
        if excludedIds[i] in similarMangaIdCounts:
            # print(excludedIds[i], similarMangaIdCounts[excludedIds[i]])
            del similarMangaIdCounts[excludedIds[i]]
    # filter out manga the user has already interacted with
    for i in range(len(ratedMangaIds)):
        if ratedMangaIds[i] in similarMangaIdCounts:
            # print(ratedMangaIds[i], similarMangaIdCounts[ratedMangaIds[i]])
            del similarMangaIdCounts[ratedMangaIds[i]]
    # print(similarMangaIdCounts)

    similarUsersManga = []
    for mangaId in similarMangaIdCounts.keys():
        if len(similarUsersManga) >= 25:  # *********** may need to change to change size of result set ***********
            break
        for j in manga:
            if mangaId == j[0]:  # I should check if content filtering already includes this manga but I dont yet
                #print(similarMangaIdCounts[mangaId], j)  # HELPFUL
                similarUsersManga.append(j)
    #print('\n'.join([str(x) for x in similarUsersManga]))
    #print("********************************************************************************")
    #print([x[0] for x in similarUsersManga])
    # *********** change number to change size of result set ***********
    recommendedManga = [i for sub in itertools.zip_longest(similarUsersManga, similarManga) for i in sub][0:50]
    # above line creates recommended manga by alternating between manga in each list
    #print('\n'.join([str(x) for x in recommendedManga]))  # HELPFUL

    # return list of json with manga info for the highest scored recommendations
    results = []
    for x in recommendedManga:
        results.append({"id": x[0], "title": x[2], "pictureLink": x[9]})
    return json.dumps(results)


callFromNode = True
includeAll = [[1, 27691], [1946, 2022], [1, 6477],
              [False] * 4, [False] * 18, [False] * 51, [False] * 5, [False] * 18, [False] * 51, [False] * 5]
noAdventure = "[[1, 27691],[1946, 2022],[1, 6477],[false,false,false,false],[true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false]]"
jsonTestFilter = "[[1,27691],[1946,1999],[1,6477],[false,true,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false]]"
jsonTestFilter2 = "[[1, 1000],[2005, 2011],[1, 33],[false,true,false,false],[true,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false]]"
jsonTestFilter3 = "[[1,27691],[1946,1999],[1,6477],[false,false,false,false],[false,true,false,true,true,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false],[false,false,false,false,false]]"
if callFromNode:
    uId = int(sys.argv[1])
    filtersIn = sys.argv[2]
    filtersIn = json.loads(filtersIn)

    print(recommend(uId, filtersIn))
    sys.stdout.flush()
else:
    # print(recommend(10, testFilter))
    start_time = time.time()
    # print(recommend(17441, json.loads(noAdventure)))  # me
    # test with uid1, uid2, uid3, uid2768, uid10, uid17441
    print(recommend(17441, includeAll))
    print("total run time:", time.time() - start_time)
    # print(recommend(1, testFilter))
