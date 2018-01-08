bottom_categories = {
    ('Women\'s Fashion', 'Clothing', 'Jeans'),
    ('Women\'s Fashion',
     'Jeans', ),
    ('Women\'s Fashion',
     'Skirts', ),
    ('Women\'s Fashion',
     'Pants', ),
    ('Women\'s Fashion', 'Activewear', 'Activewear Pants'),
    ('Women\'s Fashion', 'Activewear', 'Activewear Shorts'),
    ('Women\'s Fashion', 'Clothing', 'Pants'),
    ('Women\'s Fashion', 'Clothing', 'Shorts'),
    ('Women\'s Fashion', 'Clothing', 'Skirts'),
}

top_categories = {
    ('Women\'s Fashion', 'Clothing', 'Dresses'),
    ('Women\'s Fashion', 'Clothing', 'Tops'),
    ('Women\'s Fashion', 'Activewear', 'Activewear Jackets'),
    ('Women\'s Fashion', 'Activewear', 'Activewear Tops'),
    ('Women\'s Fashion',
     'Dresses', ),
    ('Women\'s Fashion',
     'Jackets', ),
    ('Women\'s Fashion',
     'Sweatshirts & Hoodies', ),
    ('Women\'s Fashion',
     'Tops', ),
}

shoe_categories = {
    ('Women\'s Fashion',
     'Boots', ),
    ('Women\'s Fashion',
     'Shoes', ),
    ('Women\'s Fashion',
     'Sandals', ),
}

all_categories = bottom_categories | top_categories | shoe_categories

max_depth = max([
    len(cate) for cate in top_categories | bottom_categories | shoe_categories
])
min_depth = min([
    len(cate) for cate in top_categories | bottom_categories | shoe_categories
])


def belongs_to(category, categories):
    return first_category(category, categories) is not None


def first_category(category, categories):
    if category:
        for length in range(min_depth, max_depth + 1):
            sub_category = tuple(category[:length])
            if sub_category in categories:
                return sub_category
    return None
