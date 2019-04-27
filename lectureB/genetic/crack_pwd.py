import random
import string

password = 'brad1234'
min_len = 2
max_len = 10
# 엄마도 기억력의 한계가 있어서 비밀번호의 최대, 최소 자릿수를 지정해놓았다

def generate_word(length):
    result = ''
    x = ''.join(random.sample(string.ascii_letters + string.digits, k=length))
    return x

# 옛날 옛적 빵형이 지구에 빛이 있으라하여 첫번째 세대가 태어났다
def generate_population(size, min_len, max_len):
    population = []
    for i in range(size):
        # generate words with balanced length
        length = i % (max_len - min_len + 1) + min_len
        population.append(generate_word(length))
    return population

print(generate_word(length=10))
pop = generate_population(size=100, min_len=min_len, max_len=max_len)
pop


def fitness(password, test_word):
    score = 0

    if len(password) != len(test_word):
        return score

    # if fit length, I'll give you score 0.5
    len_score = 0.5
    score += len_score

    for i in range(len(password)):
        if password[i] == test_word[i]:
            score += 1

    # 백점 만점에 몇점?
    return score / (len(password) + len_score) * 100


fitness('abcde', 'abcde')


def compute_performace(population, password):
    performance_list = []
    for individual in population:
        score = fitness(password, individual)

        # we can predict length of password
        if score > 0:
            pred_len = len(individual)
        performance_list.append([individual, score])

    population_sorted = sorted(performance_list, key=lambda x: x[1], reverse=True)
    return population_sorted, pred_len


def select_survivors(population_sorted, best_sample, lucky_few, password_len):
    next_generation = []

    for i in range(best_sample):
        if population_sorted[i][1] > 0:
            next_generation.append(population_sorted[i][0])

    lucky_survivors = random.sample(population_sorted, k=lucky_few)
    for l in lucky_survivors:
        next_generation.append(l[0])

    # generate new population if next_generation is too small
    while len(next_generation) < best_sample + lucky_few:
        next_generation.append(generate_word(length=password_len))

    random.shuffle(next_generation)
    return next_generation


pop_sorted, pred_len = compute_performace(pop, password)
survivors = select_survivors(pop_sorted, best_sample=20, lucky_few=20, password_len=pred_len)

print('Password length must be %s' % pred_len)
print('survivors=', survivors)



def create_child(individual1, individual2):
    child = ''
    min_len_ind = min(len(individual1), len(individual2))
    for i in range(min_len_ind):
        if (int(100 * random.random()) < 50):
            child += individual1[i]
        else:
            child += individual2[i]
    return child

# 우리는 그렇게 자비롭지 않기때문에 부모를 내 맘대로 짝지어줄 것이다
def create_children(parents, n_child):
    next_population = []
    for i in range(int(len(parents)/2)):
        for j in range(n_child):
            next_population.append(create_child(parents[i], parents[len(parents) - 1 - i]))
    return next_population

children = create_children(survivors, 5)

print('children=', children)


def mutate_word(word):
    idx = int(random.random() * len(word))
    if (idx == 0):
        word = random.choice(string.ascii_letters + string.digits) + word[1:]
    else:
        word = word[:idx] + random.choice(string.ascii_letters + string.digits) + word[idx+1:]
    return word

def mutate_population(population, chance_of_mutation):
    for i in range(len(population)):
        if random.random() * 100 < chance_of_mutation:
            population[i] = mutate_word(population[i])
    return population

new_generation = mutate_population(population=children, chance_of_mutation=10)

print('new_generation=', new_generation)

password = 'bBanGhyONg'
n_generation = 300
population = 100
best_sample = 20
lucky_few = 20
n_child = 5
chance_of_mutation = 10

pop = generate_population(size=population, min_len=min_len, max_len=max_len)

for g in range(n_generation):
    pop_sorted, pred_len = compute_performace(population=pop, password=password)

    if int(pop_sorted[0][1]) == 100:
        print('SUCCESS! The password is %s' % (pop_sorted[0][0]))
        break

    survivors = select_survivors(population_sorted=pop_sorted, best_sample=best_sample, lucky_few=lucky_few,
                                 password_len=pred_len)

    children = create_children(parents=survivors, n_child=n_child)

    new_generation = mutate_population(population=children, chance_of_mutation=10)

    pop = new_generation

    print('===== %sth Generation =====' % (g + 1))
    print(pop_sorted[0])

