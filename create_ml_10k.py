import argparse
import os
import random

from utility.load_data import process_data_ml_1m


def read_records(path):
    with open(path, encoding='latin-1') as source:
        return [line.rstrip('\r\n').split('::') for line in source if line.strip()]


def create_subset(input_dir, output_dir, target_ratings=10000, seed=1):
    if target_ratings < 2:
        raise ValueError('target_ratings must be at least 2.')
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            'Output directory is not empty: {}. Choose another directory or move its contents first.'.format(
                output_dir
            )
        )

    ratings = read_records(os.path.join(input_dir, 'ratings.dat'))
    users = {int(row[0]): row for row in read_records(os.path.join(input_dir, 'users.dat'))}
    movies = {int(row[0]): row for row in read_records(os.path.join(input_dir, 'movies.dat'))}
    shuffled_ratings = ratings.copy()
    random.Random(seed).shuffle(shuffled_ratings)

    user_counts = {}
    eligible_rating_count = 0
    sampled_ratings = []
    for row in shuffled_ratings:
        user_id = int(row[0])
        previous_count = user_counts.get(user_id, 0)
        user_counts[user_id] = previous_count + 1
        if previous_count == 1:
            eligible_rating_count += 2
        elif previous_count >= 2:
            eligible_rating_count += 1
        sampled_ratings.append(row)
        if eligible_rating_count >= target_ratings:
            break

    selected_users = {
        user_id for user_id, count in user_counts.items() if count >= 2
    }
    selected_ratings = [
        row for row in sampled_ratings if int(row[0]) in selected_users
    ]
    if len(selected_ratings) < target_ratings:
        raise ValueError(
            'Could not select {} ratings with at least two ratings per user; selected {}.'.format(
                target_ratings, len(selected_ratings)
            )
        )

    selected_movies = {int(row[1]) for row in selected_ratings}
    user_ids = sorted(selected_users)
    movie_ids = sorted(selected_movies)
    user_map = {old_id: new_id for new_id, old_id in enumerate(user_ids, start=1)}
    movie_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids, start=1)}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ratings.dat'), 'w', encoding='latin-1', newline='\n') as out:
        for row in sorted(
            selected_ratings,
            key=lambda record: (user_map[int(record[0])], int(record[3]), int(record[1]))
        ):
            out.write('{}::{}::{}::{}\n'.format(
                user_map[int(row[0])], movie_map[int(row[1])], row[2], row[3]
            ))

    with open(os.path.join(output_dir, 'users.dat'), 'w', encoding='latin-1', newline='\n') as out:
        for old_id in user_ids:
            row = users[old_id]
            out.write('::'.join([str(user_map[old_id])] + row[1:]) + '\n')

    with open(os.path.join(output_dir, 'movies.dat'), 'w', encoding='latin-1', newline='\n') as out:
        for old_id in movie_ids:
            row = movies[old_id]
            out.write('::'.join([str(movie_map[old_id])] + row[1:]) + '\n')

    process_data_ml_1m(output_dir)
    return len(selected_ratings), len(user_ids), len(movie_ids)


def main():
    parser = argparse.ArgumentParser(
        description='Create a MovieLens-format subset with about 10,000 ratings.'
    )
    parser.add_argument('--input_dir', default='data/recommendation/ml-1m')
    parser.add_argument('--output_dir', default='data/recommendation/ml-10k')
    parser.add_argument('--target_ratings', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    rating_count, user_count, movie_count = create_subset(
        args.input_dir, args.output_dir, args.target_ratings, args.seed
    )
    print('Created {} with {} ratings, {} users, and {} movies.'.format(
        args.output_dir, rating_count, user_count, movie_count
    ))
    print('Each selected user has at least two ratings; train.txt/test.txt use the project split format.')


if __name__ == '__main__':
    main()
